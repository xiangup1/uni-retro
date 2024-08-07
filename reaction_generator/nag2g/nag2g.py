import ast
import json
import math
import os
from argparse import Namespace

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from unicore import checkpoint_utils, distributed_utils, tasks, utils
from unicore.logging import progress_bar
from unicore.utils import import_user_module

from reaction_generator import Reaction, ReactionGenerator

from .search_strategies.score_beam_search_generator import \
    SequenceScoreGeneratorBeamSearch


def setnomap2smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    mol = AllChem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)   

def setmap2smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    mol = AllChem.RemoveHs(mol)
    [atom.SetAtomMapNum(idx + 1) for idx, atom in enumerate(mol.GetAtoms())]
    return Chem.MolToSmiles(mol)

def empty_func(*args, **kwargs):
    pass

class NAG2GGenerator(ReactionGenerator):
    def __init__(self, args:Namespace):
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        distributed_utils.call_main(args, empty_func)
        import_user_module(args)

        use_fp16 = args.fp16
        use_cuda = torch.cuda.is_available() and not args.cpu

        if use_cuda:
            torch.cuda.set_device(args.device_id)

        if args.distributed_world_size > 1:
            data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
            data_parallel_rank = distributed_utils.get_data_parallel_rank()
        else:
            data_parallel_world_size = 1
            data_parallel_rank = 0

        overrides = ast.literal_eval(args.model_overrides)

        state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
        task = tasks.setup_task(args)
        model = task.build_model(args)

        model.load_state_dict(state["model"], strict=False)
        if use_fp16:
            model = model.half()
        if use_cuda:
            model.cuda()

        # Build loss
        loss = task.build_loss(args)
        loss.eval()
        model.eval()

        np.random.seed(args.seed)

        if args.search_strategies == "SequenceScoreGeneratorBeamSearch":
            generator = SequenceScoreGeneratorBeamSearch(
                [model],
                task.dictionary,
                beam_size=args.beam_size,
                len_penalty=args.len_penalty,
                max_len_b=511,
                unk_penalty=math.inf,
            )
        
        self.args, self.use_cuda, self.task, self.generator, self.data_parallel_world_size, self.data_parallel_rank = args, use_cuda, task, generator, data_parallel_world_size, data_parallel_rank

    def generate(self, product):
        return self.run_reverse_batch([product])[product]

    def run_reverse_batch(self, products):
        return_values = {}
        for i in range(len(products)-1,-1,-1):
            if len(products[i]) < 5:
                return_values[products[i]] = []
                products.pop(i)
        if not products:
            return return_values
        args, use_cuda, task, generator, data_parallel_world_size, data_parallel_rank = self.args, self.use_cuda, self.task, self.generator, self.data_parallel_world_size, self.data_parallel_rank
        mapping_products = [setmap2smiles(product) for product in products]
        name = 'product_smiles'
        dataset_empty = task.load_empty_dataset(
            name=name, init_values=mapping_products)
        dataset = task.dataset("test")
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=len(mapping_products),  # args.batch_size,
            ignore_invalid_inputs=True,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)


        uni_result = set()
        for i, sample in enumerate(itr):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            results = task.infer_step(sample, generator)
            for product, result in zip(products, results):
                return_value = []
                for smiles, score in result:
                    try:
                        parts = []
                        for part in smiles.split('.'):
                            parts.append(setnomap2smiles(Chem.MolToSmiles(Chem.MolFromSmiles(part))))
                            assert parts[-1]
                            assert '>' not in parts[-1] and '<' not in parts[-1]
                        rxn = Reaction(product, parts)
                        if rxn.smiles not in uni_result:
                            uni_result.add(rxn.smiles)
                            return_value.append(rxn)
                    except Exception as inst:
                        pass
                return_values[product] = return_value
        return return_values

if __name__ == '__main__':
    path = 'resource/nag2g'
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    generator = NAG2GGenerator(args)
    retro_smiles = "CCCCC1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"
    reactions = generator.generate(retro_smiles)
    for reaction in reactions:
        print(reaction.smiles)
