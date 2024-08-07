import json

import numpy as np
from IPython import embed
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      IndexType, Milvus, Status, connections, utility)
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.template_extractor import extract_from_reaction

def remove_mapping(smiles):
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

class Reaction_Milvus:

    collection_name = 'reaction_product_fingerprint'
    _DIM = 2048
    _INDEX_FILE_SIZE = 32

    @classmethod
    def to_binary_vectors(cls, vectors):
        return_value = []
        for vector in vectors:
            new_vector = []
            for i in range(len(vector) // 8):
                d = 0
                for j in range(8):
                    d = d * 2 + vector[i * 8 + j]
                new_vector.append(d)
            data = np.array(new_vector, dtype='uint8').astype("uint8")
            return_value.append(bytes(data))
        return return_value

    def __init__(self, host = None, port = None, uri = None):
        if host is not None and port is not None:
            connections.connect(host=host, port=port)
        else:
            raise Exception('Need host or uri')
        ok = utility.has_collection(Reaction_Milvus.collection_name)
        if not ok:
            rxn_id = FieldSchema(
                    name="rxn_id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    )
            product = FieldSchema(
                    name='product',
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    )
            precursors = FieldSchema(
                    name='precursors',
                    dtype=DataType.VARCHAR,
                    max_length=400,
                    )
            template = FieldSchema(
                    name='template',
                    dtype=DataType.VARCHAR,
                    max_length=800,
                    )
            fingerprint = FieldSchema(
                    name='fingerprint',
                    dtype=DataType.BINARY_VECTOR,
                    dim=Reaction_Milvus._DIM,
                    )
            schema = CollectionSchema(
                    fields=[rxn_id, fingerprint, product, precursors, template],
                    descrption='reaction search',
                    enable_dynamic_field=True
                    )
            Collection(name=Reaction_Milvus.collection_name,
                    schema=schema,
                    using='default',
                    shards_num=2
                    )
        self.collection = Collection(Reaction_Milvus.collection_name)

    def loadData(self, file_names):
        size = self.collection.num_entities
        input = {'rxn_id':[], 'fingerprint':[], 'product':[], 'precursors':[], 'template':[]}
        for file_name in file_names:
            with open(file_name, 'r') as f:
                while True:
                    line = f.readline()
                    smiles = line.split('\t')[0]
                    try:
                        if '>' in smiles:
                            precursors = smiles.split('>')[0]
                            products = smiles.split('>')[2]
                            for product in products.split('.'):
                                template = extract_from_reaction({'_id':None, 'reactants':precursors, 'products':product})
                                if template and 'reaction_smarts' in template:
                                    mol = Chem.MolFromSmiles(product)
                                    if len(mol.GetAtoms()) > 2:                                    
                                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, Reaction_Milvus._DIM)
                                        input['rxn_id'].append(size+len(input['rxn_id']))
                                        input['fingerprint'].append([int(ch) for ch in fp.ToBitString()])
                                        input['product'].append(remove_mapping(product))
                                        input['precursors'].append(remove_mapping(precursors))
                                        input['template'].append(template['reaction_smarts'])
                    except Exception as inst:
                        pass
                    if len(input['rxn_id']) >= 10000 or not line:
                        data = [input['rxn_id'], self.to_binary_vectors(input['fingerprint']), input['product'], input['precursors'], input['template']]
        
                        self.collection.insert(data)
                        self.collection.flush()
                        index_params = {
                                "metric_type":"JACCARD",
                                "index_type":"BIN_FLAT",
                                "params":{"nlist":16384}
                                }
                        self.collection.create_index(
                                field_name='fingerprint',
                                index_params=index_params,
                                )
                        size += len(input['rxn_id'])
                        input = {'rxn_id':[], 'fingerprint':[], 'product':[], 'precursors':[], 'template':[]}
                    if not line:
                        break
        self.collection.flush()
        self.collection.create_index(
            field_name='fingerprint',
            index_params=index_params,
            )
        print(utility.index_building_progress(Reaction_Milvus.collection_name))

    def query(self, queries, topk = 500):
        return_value = []
        search_param = {"nprobe": 128}
        query_vectors = []
        for i, query in enumerate(queries):
            mol = Chem.MolFromSmiles(query)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, Reaction_Milvus._DIM)
            query_vectors.append([int(ch) for ch in fp.ToBitString()])
            if i == len(queries) - 1 or len(query_vectors) % 1000 == 0:
                vectors = np.asarray(query_vectors, dtype=int)
                param = {
                    'collection_name': Reaction_Milvus.collection_name,
                    'query_records': self.to_binary_vectors(query_vectors),
                    'top_k': topk,
                    'params': search_param,
                }
                search_params = {
                        "metric_type": "JACCARD",
                        "params": {"nprobe": 10},
                        "offset": 0,
                        }
                results = self.collection.search(
                        data=self.to_binary_vectors(query_vectors),
                        anns_field='fingerprint',
                        param=search_params,
                        limit=topk,
                        expr=None,
                        output_fields=['rxn_id','template','product','precursors'],
                        #consistency_level="Strong",
                        )
                return_value.extend(results)
                query_vectors = []
        return return_value

    def drop(self,):
        utility.drop_collection(Reaction_Milvus.collection_name)


