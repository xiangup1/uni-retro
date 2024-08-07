import os
import json
from importlib import import_module
from argparse import Namespace
from typing import List
from reaction_generator.basic import Reaction

class ReactionChecker:
    def __init__(self, args:Namespace):
        pass

    def check(self, reactions:List[Reaction]) -> List[Reaction]:
        pass

def loadChecker(path):
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    module = import_module(infor['module_path'])
    checker = getattr(module, infor['checker'])(args)
    return checker

if __name__ == "__main__":
    checker = loadChecker("resource/bert_checker")
    from reaction_generator.basic import Reaction
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
    
    result = checker.check(rxns)

    for reaction in result:
        print(reaction.smiles)