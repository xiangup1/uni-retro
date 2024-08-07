from rdkit import Chem

def simplify(smiles):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    except Exception:
        return smiles
