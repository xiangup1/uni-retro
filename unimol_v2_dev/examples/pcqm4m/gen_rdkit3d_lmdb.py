import gzip
import os, sys
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem

lines = gzip.open("data.csv.gz", "r").readlines()

smiles = []

for i in range(1, len(lines)):
    s = lines[i].decode().split(",")[1]
    smiles.append(s)


def process_one(i):
    try:
        cur_smile = smiles[i]
        m = Chem.MolFromSmiles(cur_smile)
        m2 = Chem.AddHs(m)
        AllChem.EmbedMolecule(m2)
        AllChem.MMFFOptimizeMolecule(m2)
        content = Chem.MolToMolBlock(m2)
        return gzip.compress(pickle.dumps(content))
    except:
        return None


os.system("rm -f init_3D_rdkit.lmdb")

env_new = lmdb.open(
    "init_3D_rdkit.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_new.begin(write=True)
i = 0
with Pool(96) as pool:
    for ret in tqdm(pool.imap(process_one, range(len(smiles))), total=len(smiles)):
        txn_write.put(i.to_bytes(4, byteorder="big"), ret)
        # use `int.from_bytes(key, "big")` to decode from bytes
        i += 1
        if i % 10000 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)

txn_write.commit()
env_new.close()
