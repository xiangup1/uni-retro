import json
import os
from argparse import Namespace

from rdkit import Chem 
from rdkit.Chem import AllChem
from sshtunnel import SSHTunnelForwarder

from reaction_generator.basic import Reaction, ReactionGenerator
from reaction_generator.similarity.reaction_milvus import Reaction_Milvus


class SimilarityGenerator(ReactionGenerator):
    def __init__(self, args:Namespace):
        self.milvus_server = SSHTunnelForwarder(
            ssh_address_or_host = (args.milvus_address, 22), 
            ssh_username = args.milvus_username,
            ssh_password = args.milvus_password,
            remote_bind_address=(args.milvus_remote_address, args.milvus_remote_port)
            )
        self.milvus_server.start()
        self.milvus = Reaction_Milvus(args.milvus_db_address, self.milvus_server.local_bind_port)
        self.milvus.collection.load()
        self.topk = args.topk

    def generate(self, product):
        return self.run_reverse_batch([product])[product]

    def run_reverse_batch(self, products):
        results = self.milvus.query(products)
        return_values = {}
        for product, query in zip(products, results):
            return_value = []
            reactions = set()
            mol = Chem.MolFromSmiles(product)
            for result in query:
                distance = result.distance
                template = result.entity.get('template')
                reagents = result.entity.get('reagent').split('.')
                precursors = result.entity.get('precursors')
                if result.entity.get('product') == product:
                    precursors = precursors.split('.')
                    smiles = '.'.join(sorted(precursors))
                    if smiles not in reactions:
                        return_value.append(
                            Reaction(product,result) 
                            )
                        reactions.add(smiles)
                elif template:
                    rxn = AllChem.ReactionFromSmarts(template)
                    plans = rxn.RunReactants([mol])
                    for precursors in plans:
                            result = []
                            for precursor in precursors:
                                result.append(Chem.MolToSmiles(precursor))
                            smiles = '.'.join(sorted(result))
                            if not Chem.MolFromSmiles(smiles):
                                continue
                            if smiles not in reactions:
                                return_value.append(
                                    Reaction(product, result)
                                    )
                                reactions.add(smiles) 
                if len(return_value) >= self.topk:
                    break
            return_values[product] = return_value
        return return_values

if __name__ == '__main__':
    path = 'resource/similarity'
    with open(os.path.join(path, 'infor.json'), 'r') as f:
        infor = json.load(f)
    args = Namespace(**infor['args'])
    generator = SimilarityGenerator(args)
    
    result = generator.generate('CCOC(=O)/C(=N\\OC(C)(C)C(=O)OC(C)(C)C)c1csc(NC(=O)OC(C)(C)C)n1')
    print(result)

