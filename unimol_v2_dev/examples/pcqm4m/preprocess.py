import os
import sys
import json
import glob
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
import contextlib
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
import copy

M = 1000
N = 100

def single_conf_gen(tgt_mol, num_confs=1000, FF=None):
    mol = Chem.AddHs(tgt_mol)
    allconformers = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, randomSeed=42, clearConfs=True)
    sz = len(allconformers)
    for i in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i)
        except:
            continue
    mol = Chem.RemoveHs(mol)
    return mol

def clustering(mol, M=1000, N=100):
    rdkit_mol = single_conf_gen(mol, num_confs=M)
    sz = len(rdkit_mol.GetConformers())
    tgt_coords = rdkit_mol.GetConformers()[0].GetPositions().astype(np.float16)
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)
    rdkit_coords_list = []
    for i in range(sz):
        _coords = rdkit_mol.GetConformers()[i].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)   # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))

    rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(sz,-1)
    cluster_size = min(N, sz)
    ids = KMeans(n_clusters=cluster_size, random_state=42).fit_predict(rdkit_coords_flatten).tolist()
    coords_list = [rdkit_coords_list[ids.index(i)] for i in range(cluster_size)]
    if len(coords_list) < N:
        coords_list += (N - len(coords_list)) * [coords_list[-1]]
    return coords_list

def single_process(content):
    smi, mol, e_gap = content[0], content[1], content[2]
    tgt_mol = copy.deepcopy(mol)
    tgt_mol = Chem.RemoveHs(tgt_mol)
    rdkit_cluster_coords_list = clustering(tgt_mol, M=M, N=N)
    atoms = [atom.GetSymbol() for atom in tgt_mol.GetAtoms()]
    sz = len(rdkit_cluster_coords_list)
    assert sz == N
    ## check target molecule atoms is the same as the input molecule
    for _mol in tgt_mol_list:
        _mol = Chem.RemoveHs(_mol)
        _atoms = [atom.GetSymbol() for atom in _mol.GetAtoms()]
        assert _atoms == atoms, print(smi)

    tgt_coords_list, rdkit_coords_list, target_rmsd_list  = [], [], []

    tgt_coords = tgt_mol.GetConformer().GetPositions().astype(np.float32)
    tgt_coords = tgt_coords - tgt_coords.mean(axis=0)
    tgt_coords_list = [tgt_coords]

    for i in range(sz):
        _coords = rdkit_cluster_coords_list[i]
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()).astype(np.float32))
        target_rmsd_list.append(_score)

    return pickle.dump(
        {
            'atoms': atoms,
            'coordinates': rdkit_coords_list,
            'target_coordinates': target_coordinate_list,
            'target_rmsd': target_rmsd_list,
            'target_e_gap': e_gap,
            'id': smi,
            'can_smi': smi,
        }
    )
    return dump

def write_lmdb(content_list, output_dir, name, nthreads=16):

        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir, f'{name}_{M}_{N}.lmdb')
        print(output_name)
        try:
            os.remove(output_name)
        except:
            pass
        env_new = lmdb.open(
            output_name,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(100e9),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            for inner_output in tqdm(pool.imap(inner_process, content_list)):
                if inner_output is not None:
                    txn_write.put(f'{i}'.encode("ascii"), pickle.dumps(inner_output, protocol=-1))
                    i += 1
                    if i % 100 == 0:
                        print('{} process {} lines'.format(output_name, i))
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(output_name, i))
            txn_write.commit()
            env_new.close()

def inner_process(content):
    try:
        return single_process(content)
    except:
        return None

def prepare_data():
    suppl = Chem.SDMolSupplier('pcqm4m-v2-train.sdf')
    df = pd.read_csv('data.csv.gz', index_col=0)
    split = torch.load('split_dict.pt')

    ### collect training 3D conformers
    smiles_list, mol_list = [], []
    for mol in tqdm(suppl):
        smiles_list.append(Chem.MolToSmiles(mol))
        mol_list.append(mol)
    df_mol = pd.DataFrame({'smiles':smiles_list, 'mol':mol_list})

    ### get train, val, test idx
    train_idx, valid_idx, test_dev_idx, test_challenge_idx = split['train'], split['valid'], split['test-dev'], split['test-challenge']
    df['FLAG'] = -1
    df.iloc[train_idx, 'FLAG'] = 1
    df.iloc[valid_idx, 'FLAG'] = 2
    df.iloc[test_dev_idx, 'FLAG'] = 3
    df.iloc[test_challenge_idx, 'FLAG'] = 4

    ### merge 3D conformers with smi data
    df = pd.merge(df, pd.DataFrame(df_mol), on='smiles', how='left')

    print(df['FLAG'].value_counts())
    print(df['mol'].notnull().sum())


    df.iloc[~train_idx, 'mol'] = df.iloc[~train_idx, 'smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    ### TO DO ###
    ### recheck prepare data
    ###

    return df

def check(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))
    cnt = 0
    for key in tqdm(_keys):
        datapoint_pickled = txn.get(key)
        data = pickle.loads(datapoint_pickled)
        if data['atoms'] is None:
            print(data)
            cnt += 1
        # if len(data['coordinates']) != len(data['target_coordinates']):
        #     cnt += 1
    print(cnt)
    env.close()

def write_v2(lmdb_inpath, lmdb_outpath):
    env = lmdb.open(
        lmdb_inpath,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    _keys = list(txn.cursor().iternext(values=False))

    env_new = lmdb.open(
        lmdb_outpath,
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
    for key in tqdm(_keys):
        datapoint_pickled = txn.get(key)
        data = pickle.loads(datapoint_pickled)
        if data['atoms'] is not None and len(data['atoms'])>=1:
            txn_write.put(key, pickle.dumps(data, protocol=-1))
            i += 1
        else:
            print('miss shape size: ', key)
        if i % 10000 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)
    print("total: ", i)
    txn_write.commit()
    env_new.close()
    env.close()

if __name__ == '__main__':
    # prepare_data()
    # check('train.lmdb')
    write_v2('train.lmdb', 'train_v2.lmdb')