import pickle
from pipeline_utils import simplify
from typing import Dict

def loadBuildingBlock(path:str)->Dict[str, float]:
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            bb = pickle.load(f)
    else:
        bb = {}
        with open(path,'r') as f:
            for line in f.readlines():
                for part in line.replace('\n','').split(' ')[0].split('.'):
                    bb[simplify(part)] = 0
    return bb
