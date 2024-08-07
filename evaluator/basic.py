import os
import json
from importlib import import_module
from argparse import Namespace
from typing import List

class Evaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, products:List[str]) -> List[float]:
        pass

def loadEvaluator(path):
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    module = import_module(infor['module_path'])
    evaluator = getattr(module, infor['evaluator'])(args)
    return evaluator

if __name__ == "__main__":
    evaluator = loadEvaluator("resource/bert_evaluator")
    smiles = ["CCCCC1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"]
    print(evaluator.evaluate(smiles))
