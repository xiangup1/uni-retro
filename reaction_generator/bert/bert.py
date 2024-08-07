import ast
import json
import math
import os
from argparse import Namespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from unicore import checkpoint_utils, distributed_utils, tasks, utils
from unicore.logging import metrics, progress_bar
from unicore.utils import import_user_module

from reaction_generator.basic import Reaction, ReactionGenerator

from .modules.score_beam_search_generator import \
    SequenceScoreGeneratorBeamSearch


class BertGenerator(ReactionGenerator):
    def __init__(self, args:Namespace):
        import_user_module(args)
        self.args = args
        assert (
            args.batch_size is not None
        ), "Must specify batch size either with --batch-size"

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

        loss = task.build_loss(args)
        loss.eval()
        model.eval()

        np.random.seed(args.seed)
        
        if args.search_strategies == "SequenceScoreBeamGenerator":
            generator = SequenceScoreGeneratorBeamSearch(
                [model],
                task.dictionary,
                beam_size=args.beam_size
            )
        self.use_cuda, self.task, self.generator = use_cuda, task, generator

    def generate(self, product:str) -> List[Reaction]:
        reply = []
        results = self.translate([product])
        for row in results.iterrows():
            infor = row[1].to_dict()
            precursors = infor['pred']
            try:
                mols = Chem.MolFromSmiles(precursors)
                precursors = Chem.MolToSmiles(mols)
            except Exception as inst:
                continue
            confidence_score = math.exp(infor['pre_score'])
            reply.append(Reaction(product, precursors.split('.')))
        return reply

    def run_reverse_batch(self, products:List[str]) -> Dict[str, List[Reaction]]:
        reply = {product:[] for product in products}
        for i in range(len(products)-1, -1, -1):
            if len(products[i]) > 120:
                products.pop(i)
        if not products:
            return reply
        results = self.translate(products)
        uni_result = set()
        for row in results.iterrows():
            infor = row[1].to_dict()
            product = infor['target']
            precursors = infor['pred']
            if product in reply:
                try:
                    mols = Chem.MolFromSmiles(precursors)
                    precursors = Chem.MolToSmiles(mols)
                except Exception as inst:
                    continue
            else:
                continue
            confidence_score = float(infor['pre_score'])
            rxn = Reaction(product, precursors.split('.'))
            if rxn.smiles not in uni_result:
                reply[product].append(rxn)
                uni_result.add(rxn.smiles)
        return reply

    def translate(self, smiles, seed=42):
        args = self.args
        if args.distributed_world_size > 1:
            data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
            data_parallel_rank = distributed_utils.get_data_parallel_rank()
        else:
            data_parallel_world_size = 1
            data_parallel_rank = 0

        use_cuda, task, generator = self.use_cuda, self.task, self.generator
        dataset, _ = task.load_infer_dataset(smiles=smiles, seed=seed)

        # Initialize data iterator
        
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on smiles subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            if args.search_input == 'list':
                result, log_output = task.test_list_step(args, sample, generator, loss, i,args.seed)
            else:
                result = task.infer_step(args, sample, generator)
            log_outputs.append(result)

        log_output_res = pd.concat(log_outputs)
        return log_output_res

if __name__ == '__main__':
    path = 'resource/bert_generator'
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    generator = BertGenerator(args)
    retro_smiles = "CCCCC1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"
    reactions = generator.generate(retro_smiles)
    for reaction in reactions:
        print(reaction.smiles)
