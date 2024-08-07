try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import logging
import numpy as np
import re
from rdkit import Chem

logger = logging.getLogger(__name__)


def collate_list_tokens(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of (list of 1d tensors) into a padded 2d tensor."""

    size_h = len(values[0])
    size = 0
    for v in values:
        for j in v:
            size = max(size, j.size(0))

    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0][0].new(len(values), size_h, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        for j, v_j in enumerate(v):
            copy_tensor(v_j, res[i][j][:len(v_j)])
    return res


def smi_tokenizer(smi):
    """Tokenize a SMILES sequence or reaction"""
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

# canonical_smiles_with_am, remove_am_without_canonical, canonical_smiles, randomize_smiles_with_am 
# mainly from https://github.com/yuewan2/Retroformer/tree/4e94540e06f5a0ed52ac7b7c4acdd98e92aeda56/retroformer
def canonical_smiles_with_am(smi):
    """Canonicalize a SMILES with atom mapping"""
    atomIdx2am, pivot2atomIdx = {}, {}
    mol = Chem.MolFromSmiles(smi)
    atom_ordering = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atomIdx2am[atom.GetIdx()] = atom.GetProp('molAtomMapNumber')
            atom.ClearProp('molAtomMapNumber')
        else:
            atomIdx2am[atom.GetIdx()] = '0'
        atom_ordering.append(atom.GetIdx())

    unmapped_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ordering, canonical=False)
    mol = Chem.MolFromSmiles(unmapped_smi)
    cano_atom_ordering = list(Chem.CanonicalRankAtoms(mol))
    for i, j in enumerate(cano_atom_ordering):
        pivot2atomIdx[j + 1] = i
        mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', j + 1)

    new_tokens = []
    for token in smi_tokenizer(Chem.MolToSmiles(mol)):
        if re.match('.*:([0-9]+)]', token):
            pivot = re.match('.*(:[0-9]+])', token).group(1)
            token = token.replace(pivot, ':{}]'.format(atomIdx2am[pivot2atomIdx[int(pivot[1:-1])]]))
        new_tokens.append(token)

    canonical_smi = ''.join(new_tokens)
    # canonical reactants order
    if '.' in canonical_smi:
        canonical_smi_list = canonical_smi.split('.')
        canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
        canonical_smi = '.'.join(canonical_smi_list)
    return canonical_smi

def remove_am_without_canonical(smi_am, force_canonical=False):
    """Get the canonical SMILES by token modification (smiles arranged by CanonicalRankAtoms)
    :param smi_am: SMILES from `canonical_smiles_with_am`
    :param force_canonical: force the output to be canonical, not recommended since it may break the alignment
    :return:
    """

    def check_special_token(token):
        pattern = "(Mg|Zn|Si|Sn|Se|se|Ge|K|Ti|Pd|Mo|Ce|Ta|As|te|Pb|Ru|Ag|W|Pt|Co|Ca|Xe|11CH3|Rh|Tl|V|131I|Re|13c|siH|La|pH|Y|Zr|Bi|125I|Sb|Te|Ni|Fe|Mn|Cr|Al|Na|Li|Cu|nH[0-9]?|NH[1-9]?\+|\+|-|@|PH[1-9]?)"
        regex = re.compile(pattern)
        return regex.findall(token)

    new_tokens = []
    for token in smi_tokenizer(smi_am):
        # Has atommapping:
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            # print(token)
            token = token.replace(re.match('.*(:[0-9]+)]', token).group(1), '')
            explicitHs = re.match('.*(H[1-9]?).*', token)
            onlyH = re.match('\[[1-9]?H', token)
            if explicitHs and not check_special_token(token) and not onlyH:
                token = token.replace(explicitHs.group(1), '')[1:-1]
            elif not check_special_token(token) and not onlyH:
                token = token[1:-1]
            else:
                token = token

        new_tokens.append(token)
    canonical_smi = ''.join(new_tokens)

    try:
        smi1 = canonical_smiles(canonical_smi)
        mol = Chem.MolFromSmiles(canonical_smi)
        if mol is None:
            canonical_smi = clear_map_number_base(smi_am)            
    except:
        canonical_smi = clear_map_number_base(smi_am)

    if force_canonical:
        canonical_smi = canonical_smiles(canonical_smi)
    return canonical_smi

def rooted_smiles_with_am(smi, root = -1, canonical=True):
    """Rooted a SMILES with atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, rootedAtAtom=int(root), canonical=canonical)


def randomize_smiles_with_am(smi):
    """Randomize a SMILES with atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    random_root = np.random.choice([(atom.GetIdx()) for atom in mol.GetAtoms()])
    return Chem.MolToSmiles(mol, rootedAtAtom=int(random_root))

def canonical_smiles(smi):
    """Canonicalize a SMILES without atom mapping"""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    else:
        canonical_smi = Chem.MolToSmiles(mol)
        if '.' in canonical_smi:
            canonical_smi_list = canonical_smi.split('.')
            canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
            canonical_smi = '.'.join(canonical_smi_list)
        return canonical_smi

def clear_map_number_base(smi):
    """Clear the atom mapping number of a SMILES sequence"""
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, canonical = False)
    else:
        return smi

def collate_cross_2d(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 2d tensors into a padded 2d tensor."""
    size_h = max(v.size(0) for v in values)
    size_w = max(v.size(1) for v in values)
    if pad_to_multiple != 1 and size_h % pad_to_multiple != 0:
        size_h = int(((size_h - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    if pad_to_multiple != 1 and size_w % pad_to_multiple != 0:
        size_w = int(((size_w - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size_h, size_w).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size_h - v.size(0):, size_w - v.size(1):]
                    if left_pad else res[i][:v.size(0), :v.size(1)])
    return res


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    batch_size = len(values) if pad_to_bsz is None else max(len(values), pad_to_bsz)
    res = values[0].new(batch_size, size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res



def collate_tokens_coords(
    values,
    pad_idx,
    left_pad=False,
    pad_to_length=None,
    pad_to_multiple=1,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)
    res = values[0].new(len(values), size, 3).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):, :]
                    if left_pad else res[i][: len(v), :])
    return res


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.
    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def batch_by_dynamic(
    indices,
    num_tokens_fn,
    num_tokens_vec=None,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
    fixed_shapes=None,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.
    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        num_tokens_vec (List[int], optional): precomputed vector of the number
            of tokens for each index in indices (to enable faster batch generation)
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be less than N or a multiple of N (default: 1).
        fixed_shapes (List[Tuple[int, int]], optional): if given, batches will
            only be created with the given shapes. *max_sentences* and
            *required_batch_size_multiple* will be ignored (default: None).
    """

    # added int() to avoid TypeError: an integer is required
    max_tokens = int(max_tokens) if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if not isinstance(indices, np.ndarray):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    if num_tokens_vec is not None and not isinstance(num_tokens_vec, np.ndarray):
        num_tokens_vec = np.fromiter(num_tokens_vec, dtype=np.int64, count=-1)

    if fixed_shapes is None:
        if num_tokens_vec is None:
            return batch_by_size_fn(
                indices,
                num_tokens_fn,
                max_tokens,
                max_sentences,
                bsz_mult,
            )
        else:
            return batch_by_size_vec(
                indices,
                num_tokens_vec,
                max_tokens,
                max_sentences,
                bsz_mult,
            )

    else:
        fixed_shapes = np.array(fixed_shapes, dtype=np.int64)
        sort_order = np.lexsort(
            [
                fixed_shapes[:, 1].argsort(),  # length
                fixed_shapes[:, 0].argsort(),  # bsz
            ]
        )
        fixed_shapes_sorted = fixed_shapes[sort_order]
        return batch_fixed_shapes_fast(indices, num_tokens_fn, fixed_shapes_sorted)


def batch_by_size_vec(
    indices,
    num_tokens_vec,
    max_tokens,
    max_sentences,
    bsz_mult,
):
    if indices.shape[0] == 0:
        return []
    assert max_tokens <= 0 or np.max(num_tokens_vec) <= max_tokens, (
        f"Sentences lengths should not exceed max_tokens={max_tokens}"
    )

    indices_len = indices.shape[0]
    batches_ends = np.zeros(indices_len, dtype=np.int32)
    batches_ends_view = batches_ends
    num_tokens_view = num_tokens_vec

    pos = 0
    new_batch_end = 0

    new_batch_max_tokens = 0
    new_batch_sentences = 0
    new_batch_num_tokens = 0

    overflow = False
    size_matches_with_bsz_mult = False

    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        # At every pos we keep stats about the last complete batch [batch_start:batch_end),
        #      and tail [batch_end:pos].
        # 1) Every time when (batch + tail) forms a valid batch
        #      (according to max_tokens, max_sentences and bsz_mult) we append tail to batch.
        # 2) When (batch+tail) violates max_tokens or max_sentences constraints
        #      we finalize running batch, and tail becomes a new batch.
        # 3) There is a corner case when tail also violates constraints.
        #      In that situation [batch_end:pos-1] (tail without the current pos)
        #      gets added to the finalized batches, while [pos:pos] becomes a new tail.
        #
        # Important: For the sake of performance try to avoid using function calls within this loop.

        tail_max_tokens = tail_max_tokens \
            if tail_max_tokens > num_tokens_view[pos] \
            else num_tokens_view[pos]
        new_batch_end = pos + 1
        new_batch_max_tokens = batch_max_tokens \
            if batch_max_tokens > tail_max_tokens \
            else tail_max_tokens
        new_batch_sentences = new_batch_end - batch_start
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens
        overflow = (new_batch_sentences > max_sentences > 0 or
                    new_batch_num_tokens > max_tokens > 0)
        size_matches_with_bsz_mult = (new_batch_sentences < bsz_mult or
                                      new_batch_sentences % bsz_mult == 0)

        if overflow:
            tail_num_tokens = tail_max_tokens * \
                (new_batch_end - batches_ends_view[batches_count])
            tail_overflow = tail_num_tokens > max_tokens > 0
            # In case of a tail overflow finalize two batches
            if tail_overflow:
                batches_count += 1
                batches_ends_view[batches_count] = pos
                tail_max_tokens = num_tokens_view[pos]
            batch_start = batches_ends_view[batches_count]
            batches_count += 1
            new_batch_max_tokens = tail_max_tokens

        if overflow or size_matches_with_bsz_mult:
            batches_ends_view[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0
    if batches_ends_view[batches_count] != indices_len:
        batches_count += 1
    # Memory and time-efficient split
    return np.split(indices, batches_ends[:batches_count])

# 先忽略这里


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def compare_leq(a, b):
        return a <= b if not isinstance(a, tuple) else max(a) <= b

    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(
                    a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key])
                )
                for key in intersect_keys
            )
        else:
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )

    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def batch_by_size_fn(
    indices,
    num_tokens_fn,
    max_tokens,
    max_sentences,
    bsz_mult,
):
    indices_len = indices.shape[0]
    num_tokens_vec = np.zeros(indices_len, dtype=np.int64)
    indices_view = indices
    num_tokens_vec_view = num_tokens_vec
    for pos in range(indices_len):
        num_tokens_vec[pos] = num_tokens_fn(indices_view[pos])
    return batch_by_size_vec(indices, num_tokens_vec, max_tokens,
                             max_sentences, bsz_mult,)


def _find_valid_shape(
    shapes_view,
    num_sentences,
    num_tokens,
):
    """Return index of first valid shape of -1 if none is found."""
    for i in range(shapes_view.shape[0]):
        if num_sentences <= shapes_view[i][0] and num_tokens <= shapes_view[i][1]:
            return i
    return -1


def batch_fixed_shapes_fast(
    indices,
    num_tokens_fn,
    fixed_shapes_sorted,
):
    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    indices_view = indices
    shapes_view = fixed_shapes_sorted

    for i in range(len(indices_view)):
        idx = indices_view[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        shape_idx = _find_valid_shape(shapes_view, len(batch) + 1, sample_len)
        if shape_idx == -1:
            batches.append(batch)
            batch = []
            sample_lens = []
            sample_len = 0
            shapes_view = fixed_shapes_sorted
        elif shape_idx > 0:
            # small optimization for the next call to _find_valid_shape
            shapes_view = shapes_view[shape_idx:]

        batch.append(idx)

    if len(batch) > 0:
        batches.append(batch)

    return batches