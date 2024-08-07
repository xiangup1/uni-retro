import ast
import json
import os
from argparse import Namespace
from typing import List 

import numpy as np
import pandas as pd
import torch
from unicore import checkpoint_utils, distributed_utils, tasks, utils
from unicore.logging import metrics, progress_bar
from pipeline_utils import simplify

from reaction_checker import ReactionChecker
from reaction_generator.basic import Reaction


class BertChecker(ReactionChecker):

    def __init__(self, args):
        from unicore.utils import import_user_module
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

        # Build loss
        loss = task.build_loss(args)
        loss.eval()
        model.eval()

        np.random.seed(args.seed)
        # utils.set_torch_seed(args.seed)
        print('beam search params: ',args.beam_size, args.len_penalty, args.temperature)
        # generator = SimpleGeneratorBart(
        #     model,
        #     task.dictionary,
        #     beam_size=args.beam_size
        # )
        from reaction_generator.bert.modules.score_beam_search_generator import \
            SequenceScoreGeneratorBeamSearch
        
        if args.search_strategies == "SequenceScoreBeamGenerator":
            generator = SequenceScoreGeneratorBeamSearch(
                [model],
                task.dictionary,
                beam_size=args.beam_size
            )
        # generator = SequenceGeneratorBeamSearch(
        #     [model],
        #     task.dictionary,
        #     beam_size=args.beam_size
        # )

        self.use_cuda, self.task, self.generator = use_cuda, task, generator

    def check(self, reactions:List[Reaction]):
        if not reactions:
            return reactions,
        form_precursors = []
        reply = {}
        targets = []
        for idx, reaction in enumerate(reactions):
            if len('.'.join(reaction.precursors)) < 511:
                form_precursors.append('.'.join(reaction.precursors))
                targets.append(reaction.product)
                reply[reaction.smiles] = False
            else:
                reply[reaction.smiles] = False
        if form_precursors:
            result = self.translate(form_precursors)
        else:
            return []
        query_id, pred_id = 0, 0
        for row in result.iterrows():
            infor = row[1].to_dict()
            if infor['target'] == form_precursors[query_id]:
                if simplify(infor['pred']) == targets[query_id]:
                    reply[form_precursors[query_id]+'>>'+targets[query_id]] = True
            pred_id += 1
            if pred_id == self.args.beam_size:
                pred_id = 0
                query_id += 1
        return_value = [reaction for reaction in reactions if reply[reaction.smiles]]
        return return_value

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
        pp = enumerate(progress)
        for i, sample in pp:
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
    path = 'resource/bert_checker'
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    checker = BertChecker(args)
    
    reactions = [
        'COc1ccc(COC(=O)C2=C(CCl)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2ccc(C(=O)NCCN3CCCC3)c(Cl)c2OCc2ccc(OC)cc2)cc1>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CCl)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2c(Cl)ccc(C(=O)NCCN3CCCC3)c2OCc2ccc(OC)cc2)cc1>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CI)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2ccc(C(=O)NCCN3CCCC3)c(Cl)c2OCc2ccc(OC)cc2)cc1>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CCl)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2ccc(C(=O)NCCN3CCCC3)c(Cl)c2OCc2ccc(OC)cc2)cc1.[Na]I>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CCl)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2cc(Cl)c(C(=O)NCCN3CCCC3)cc2OCc2ccc(OC)cc2)cc1>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CI)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2c(Cl)ccc(C(=O)NCCN3CCCC3)c2OCc2ccc(OC)cc2)cc1>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CCl)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2c(Cl)ccc(C(=O)NCCN3CCCC3)c2OCc2ccc(OC)cc2)cc1.[Na]I>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CCl)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1.COc1ccc(COc2ccc(C(=O)NCCN3CCCC3)c(Cl)c2OCc2ccc(OC)cc2)cc1>>COc1ccc(COC(=O)C2=C(C[N+]3(CCNC(=O)c4ccc(OCc5ccc(OC)cc5)c(OCc5ccc(OC)cc5)c4Cl)CCCC3)CS(=O)[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)O)cs1.COc1ccc(COC(=O)C2=C(CCl)CSC3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)Cl)cs1.COc1ccc(COC(=O)C2=C(CCl)CSC3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)',
        'C(=O)N23)cc1CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)O[Na])cs1.COc1ccc(COC(=O)C2=C(CCl)CSC3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)[O-])cs1.COc1ccc(COC(=O)C2=C(CCl)CSC3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)C(C)(C)ON.COc1ccc(COC(=O)C2=C(CCl)CSC3C(NC(=O)C(=O)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)NC2C(=O)N3C(C(=O)O)=C(CCl)CSC23)cs1.COc1ccc(CCl)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1COc1ccc(COC(=O)C2=C(CCl)CS(=O)C3C(NC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)OS(C)(=O)=O)cs1.COc1ccc(COC(=O)C2=C(CCl)CSC3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)ON2C(=O)CCC2=O)cs1.COc1ccc(COC(=O)C2=C(CCl)CSC3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1'
        'CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)O)cs1.COc1ccc(COC(=O)C2N3C(=O)C(N)C3SCC2(O)CCl)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(/C(=N/OC(C)(C)C(=O)OC(C)(C)C)C(=O)O)cs1.COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(C(=NOC(C)(C)C(=O)OC(C)(C)C)C(=O)O)cs1.COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(/C(=N/OC(C)(C)C(=O)OC(C)(C)C)C(=O)O)cs1.COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3C(N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](N)C(=O)N23)cc1>>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(/C(=N/OC(C)(C)C(=O)OC(C)(C)C)C(=O)Cl)cs1.COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](N)C(=O)N23)cc1>CN1CCOCC1.CCOC(C)=O.c1ccncc1.ClCCl.O>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
        'CC(C)(C)OC(=O)Nc1nc(/C(=N/OC(C)(C)C(=O)OC(C)(C)C)C(=O)O)cs1.COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](N)C(=O)N23)cc1>O=P(Cl)(Cl)Oc1ccccc1.CN1CCOCC1.CCOC(C)=O.ClCCl>COc1ccc(COC(=O)C2=C(CCl)CS[C@@H]3[C@H](NC(=O)/C(=N\OC(C)(C)C(=O)OC(C)(C)C)c4csc(NC(=O)OC(C)(C)C)n4)C(=O)N23)cc1',
    ]
    
    rxns = [Reaction(reaction.split('>')[2], reaction.split('>')[0].split('.')) for reaction in reactions]
    
    reactions = checker.check(rxns)
    for reaction in reactions:
        print(reaction.smiles)
