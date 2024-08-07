import ast
import json
import os
import pickle
from argparse import Namespace

import numpy as np
import torch
from unicore import checkpoint_utils, distributed_utils, tasks, utils
from unicore.logging import progress_bar
from unicore.utils import import_user_module

from evaluator import Evaluator


class BertEvaluator(Evaluator):
    def __init__(self, args):
        import_user_module(args)
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
        self.args, self.use_cuda, self.task, self.model = args, use_cuda, task, model

    def evaluate(self, products, seed=42):
        args, use_cuda, task, generator = self.args, self.use_cuda, self.task, self.model

        if args.distributed_world_size > 1:
            data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
            data_parallel_rank = distributed_utils.get_data_parallel_rank()
        else:
            data_parallel_world_size = 1
            data_parallel_rank = 0
        dataset, _ = task.load_infer_dataset(smiles=products, seed=seed)

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
            result = task.infer_step(args, sample, generator, single_sample=False)
            log_outputs.extend(result)
        return {product:value for product, value in zip(products, log_outputs)}

if __name__ == '__main__':
    path = 'resource/bert_evaluator'
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    evaluator = BertEvaluator(args)
    input = ["NCC(=O)c1cccc(Br)c1"]
    result = evaluator.evaluate(input)
    print(result)
