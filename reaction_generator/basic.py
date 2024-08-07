import json
import os
from importlib import import_module
from typing import List
from argparse import Namespace

class Reaction:
    def __init__(self, product:str, precursors:List[str], reaction_cost=1):
        self.product = product
        self.precursors = sorted(precursors)
        self.smiles = '.'.join(self.precursors) + '>>' + self.product
        self.reaction_cost = reaction_cost

class ReactionGenerator:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, product:str) -> List[Reaction]:
        pass

def loadGenerator(path):
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    module = import_module(infor['module_path'])
    generator = getattr(module, infor['generator'])(args)
    return generator

if __name__ == "__main__":
    generator = loadGenerator("resource/bert_generator")
    for reaction in generator.generate("CCCCC1C(=O)N(c2ccccc2)N(c2ccccc2)C1=O"):
        print(reaction.smiles)
