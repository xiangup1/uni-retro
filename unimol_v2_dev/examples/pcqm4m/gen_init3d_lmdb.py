import gzip
import os, sys
import pickle5 as pickle
from tqdm import tqdm
from multiprocessing import Pool
import lmdb

lines = gzip.open("data.csv.gz", "r").readlines()

smiles = []

for i in range(1, len(lines)):
    s = lines[i].decode().split(",")[1]
    smiles.append(s)

def process_one(i):
    try:
        cur_smile = smiles[i]
        output_file = "/tmp/test_sdf/{}.sdf".format(i)
        os.system("obabel -:\"{}\" -O {} --gen3d > /dev/null 2>&1".format(cur_smile, output_file))
        content = open(output_file, "r").read()
        os.system("rm {}".format(output_file))
        return gzip.compress(pickle.dumps(content))
    except:
        return None

os.system("rm -f init_3D.lmdb")

env_new = lmdb.open(
    "init_3D.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)
txn_write = env_new.begin(write = True)
i = 0
with Pool(96) as pool:
    for ret in tqdm(pool.imap(process_one, range(len(smiles))), total=len(smiles)):
        txn_write.put(i.to_bytes(4, byteorder='big'), ret)
        # use `int.from_bytes(key, "big")` to decode from bytes
        i += 1
        if i % 10000 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)

txn_write.commit()
env_new.close()

# How to read it

# env = lmdb.open(
#     lmdb_path,
#     subdir=False,
#     readonly=True,
#     lock=False,
#     readahead=False,
#     meminit=False,
#     max_readers=256,
# )
# idx = 1024
# key = idx.to_bytes(4, byteorder='big')
# data = env.begin().get(key)
# raw_data = pickle.loads(gzip.decompress(data))

