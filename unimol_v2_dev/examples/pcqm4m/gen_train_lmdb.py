import gzip
import os, sys
import pickle
from turtle import position
from tqdm import tqdm
from multiprocessing import Pool
import lmdb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import GetBestAlignmentTransform
import numpy as np

lines = gzip.open("data.csv.gz", "r").readlines()

target = []
smiles = []

for i in range(1, len(lines)):
    try:
        s = lines[i].decode().split(",")
        smiles.append(s[1])
        target.append(float(s[2]))
    except:
        target.append(None)

del lines

init_env = lmdb.open(
    "init_3D.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)

rdkit_env = lmdb.open(
    "init_3D_rdkit.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)

label_env = lmdb.open(
    "label_3D.lmdb",
    subdir=False,
    readonly=False,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
    map_size=int(100e9),
)

with label_env.begin() as txn:
    train_keys = list(txn.cursor().iternext(values=False))


def get_info(src_mol, perm=None):
    atoms = np.array([x.GetSymbol() for x in src_mol.GetAtoms()])
    pos = src_mol.GetConformer().GetPositions()
    if perm is not None:
        new_atoms = []
        new_pos = np.zeros_like(pos)
        for i in range(len(atoms)):
            j = perm[i]
            new_atoms.append(atoms[j])
            new_pos[i, :] = pos[j, :]
        return np.array(new_atoms), new_pos
    else:
        return atoms, pos


def align_to(src_mol, ref_mol):
    t = GetBestAlignmentTransform(src_mol, ref_mol)
    perm = {x[1]: x[0] for x in t[2]}
    R = t[1][:3, :3].T
    T = t[1][:3, 3].T

    ref_atoms, ref_pos = get_info(ref_mol)
    src_atoms, src_pos = get_info(src_mol, perm)
    assert np.all(ref_atoms == src_atoms)
    src_pos = src_pos @ R + T

    def cal_rmsd(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
        sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
        msd = np.mean(sd)
        return np.sqrt(msd + eps)

    cur_rmsd = cal_rmsd(src_pos, ref_pos)
    assert np.abs(cur_rmsd - t[0]) < 1e-2
    return ref_atoms, src_pos, ref_pos


def obabel_3d_gen_by_sdf(smile, index):
    output_file = "/tmp/test_sdf/{}.sdf".format(index)
    m = Chem.MolFromSmiles(smile)
    ori_smi = Chem.MolToSmiles(m)
    content = Chem.MolToMolBlock(m)
    input_file = "/tmp/test_sdf/i_{}.sdf".format(index)
    output = open(input_file, "w")
    output.write(content)
    output.close()
    os.system(
        "obabel {} -O {} --gen3d > /dev/null 2>&1".format(input_file, output_file)
    )
    content = open(output_file, "r").read()
    os.system("rm {}".format(input_file))
    os.system("rm {}".format(output_file))
    ret = Chem.MolFromMolBlock(content)
    ret_smi = Chem.MolToSmiles(ret)
    assert ret_smi == ori_smi
    pos = ret.GetConformer().GetPositions()
    return ret


def obabel_3d_gen(smile, index):
    output_file = "/tmp/test_sdf/{}.sdf".format(index)
    os.system('obabel -:"{}" -O {} --gen3d > /dev/null 2>&1'.format(smile, output_file))
    content = open(output_file, "r").read()
    os.system("rm {}".format(output_file))
    ret = Chem.MolFromMolBlock(content)
    ret_smi = Chem.MolToSmiles(ret)
    ori_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
    pos = ret.GetConformer().GetPositions()
    if ret_smi != ori_smi:
        # if sdf cannot handle, use the smiles's result
        try:
            ret = obabel_3d_gen_by_sdf(ori_smi, index)
        except:
            pass
    return ret


def rdkit_gen(smile):
    m = Chem.MolFromSmiles(smile)
    m2 = Chem.AddHs(m)
    try:
        AllChem.EmbedMolecule(m2)
        AllChem.MMFFOptimizeMolecule(m2)
    except:
        pass
    pos = m2.GetConformer().GetPositions()
    return m2


def get_by_key(env, key):
    data = env.begin().get(key)
    if data is None:
        return data
    else:
        try:
            return pickle.loads(gzip.decompress(data))
        except:
            return None


def one_try(init_mol, label_mol):
    init_mol = Chem.RemoveHs(init_mol)
    return align_to(init_mol, label_mol)


# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    return atom_feature


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def get_graph(mol):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int32)
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)
    return x, edge_index, edge_attr


def process_one(key):
    index = int.from_bytes(key, "big")
    label_str = get_by_key(label_env, key)
    label_mol = Chem.MolFromMolBlock(label_str)
    label_mol = Chem.RemoveHs(label_mol)
    label_smi = Chem.MolToSmiles(label_mol)
    ori_smi = smiles[index]
    init_pos_list = []
    rdkit_str = get_by_key(rdkit_env, key)
    try:
        if rdkit_str is not None:
            rdkit_mol = Chem.MolFromMolBlock(rdkit_str)
            rdkit_mol = Chem.RemoveHs(rdkit_mol)
            atoms, init_pos, label_pos = align_to(init_mol, label_mol)
            init_pos_list.append(init_pos)
    except:
        pass

    init_str = get_by_key(init_env, key)

    # first try
    try:
        init_mol = Chem.MolFromMolBlock(init_str)
        atoms, init_pos, label_pos = one_try(init_mol, label_mol)
    except:
        init_mol, atoms, init_pos, label_pos = None, None, None, None

    # second try
    if init_mol is None:
        try:
            init_mol = obabel_3d_gen(ori_smi, index)
            atoms, init_pos, label_pos = one_try(init_mol, label_mol)
        except:
            init_mol, atoms, init_pos, label_pos = None, None, None, None

    # last try
    if init_mol is None:
        try:
            init_mol = obabel_3d_gen(label_smi, index)
            atoms, init_pos, label_pos = one_try(init_mol, label_mol)
        except:
            init_mol, atoms, init_pos, label_pos = None, None, None, None

    if init_pos is not None:
        init_pos_list.append(init_pos)

    if len(init_pos_list) == 0:
        try:
            init_mol = rdkit_gen(label_smi)
            atoms, init_pos, label_pos = one_try(init_mol, label_mol)
            init_pos_list.append(init_pos)
            print("rollback", index, ori_smi, label_smi)
        except:
            try:
                init_mol = Chem.MolFromSmiles(label_smi)
                AllChem.Compute2DCoords(init_mol)
                atoms, init_pos, label_pos = one_try(init_mol, label_mol)
                init_pos_list.append(init_pos)
                print("rollback to 2d", index, ori_smi, label_smi)
            except:
                print("failed", index, ori_smi, label_smi)
                return key, None

    if len(atoms) <= 0:
        print(index, label_smi)
        return key, None
    node_attr, edge_index, edge_attr = get_graph(label_mol)
    return key, gzip.compress(
        pickle.dumps(
            {
                "atoms": atoms,
                "input_pos": init_pos_list,
                "label_pos": label_pos,
                "target": target[index],
                "smi": label_smi,
                "node_attr": node_attr,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
            }
        )
    )


os.system("rm -f train.lmdb")

env_new = lmdb.open(
    "train.lmdb",
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
    for ret in tqdm(pool.imap(process_one, train_keys), total=len(train_keys)):
        key, val = ret
        if val is not None:
            txn_write.put(key, val)
        # use `int.from_bytes(key, "big")` to decode from bytes
        i += 1
        if i % 10000 == 0:
            txn_write.commit()
            txn_write = env_new.begin(write=True)

txn_write.commit()
env_new.close()
