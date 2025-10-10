from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp
from tqdm import tqdm
import random

output_file = "random_mini_db.sdf"
writer = Chem.SDWriter(output_file)
num_mols = 100
MAX_CID = 150_000_000

# Fetch first 100 CIDs from PubChem for demo
# cids = list(range(1, 2*num_mols+1))  # fetch 200 in case some fail
num_to_fetch = num_mols * 2
cids = random.sample(range(1, MAX_CID), k=num_to_fetch)

count = 0
for cid in tqdm(cids, desc="Generating molecules"):
    try:
        compound = pcp.Compound.from_cid(cid)
        smiles = compound.isomeric_smiles
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        mol.SetProp("_Name", f"PubChem_CID_{cid}")
        writer.write(mol)
        count += 1

        if count >= num_mols:
            break
    except Exception:
        continue

writer.close()
print(f"✅ Successfully generated {count} molecules into {output_file}")