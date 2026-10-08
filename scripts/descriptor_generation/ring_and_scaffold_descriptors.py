import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

def describe_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "SSSR_count": None,
            "Largest_ring_size": None,
            "Macrocycle_count": None,
            "Fraction_aromatic_rings": None,
            "Aromatic_ring_count": None,
            "Fused_ring_count": None,
            "Bridged_ring_count": None,
            "Spiro_atom_count": None,
            "Murcko_scaffold_smiles": None,
        }

    ri = mol.GetRingInfo()
    atom_rings = list(ri.AtomRings())
    sssr = ri.NumRings()
    largest_ring_size = max((len(r) for r in atom_rings), default=0)

    aromatic_ring_count = sum(
        all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        for ring in atom_rings
    )
    fraction_aromatic_rings = (
        aromatic_ring_count / sssr if sssr > 0 else 0.0
    )

    macrocycle_count = sum(1 for r in atom_rings if len(r) > 8)

    # fused rings: share ≥1 atom
    fused_flags = [any(set(r1) & set(r2) for j, r2 in enumerate(atom_rings) if i != j)
                   for i, r1 in enumerate(atom_rings)]
    fused_ring_count = sum(fused_flags)

    # bridged rings: share ≥2 atoms
    bridged_flags = [any(len(set(r1) & set(r2)) >= 2 for j, r2 in enumerate(atom_rings) if j > i)
                     for i, r1 in enumerate(atom_rings)]
    # symmetrize flags
    for i, r1 in enumerate(atom_rings):
        for j, r2 in enumerate(atom_rings):
            if j <= i: 
                continue
            if len(set(r1) & set(r2)) >= 2:
                bridged_flags[i] = True
                bridged_flags[j] = True
    bridged_ring_count = sum(bridged_flags)

    spiro_atom_count = rdMolDescriptors.CalcNumSpiroAtoms(mol)

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = (
        Chem.MolToSmiles(scaffold, isomericSmiles=False)
        if scaffold.GetNumAtoms() > 0
        else None
    )

    return {
        "SSSR_count": sssr,
        "Largest_ring_size": largest_ring_size,
        "Macrocycle_count": macrocycle_count,
        "Fraction_aromatic_rings": fraction_aromatic_rings,
        "Aromatic_ring_count": aromatic_ring_count,
        "Fused_ring_count": fused_ring_count,
        "Bridged_ring_count": bridged_ring_count,
        "Spiro_atom_count": spiro_atom_count,
        "Murcko_scaffold_smiles": scaffold_smiles,
    }

def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in df.iterrows():
        mol_id = row["identifier"]
        smi = row["canonical_smiles"]
        desc = describe_smiles(smi)
        desc["identifier"] = mol_id
        desc["canonical_smiles"] = smi
        results.append(desc)

    out_df = pd.DataFrame(results)

    # Murcko scaffold diversity: dataset-wide unique scaffolds
    unique_scaffolds = set(s for s in out_df["Murcko_scaffold_smiles"] if pd.notna(s))
    out_df.attrs["murcko_scaffold_diversity_count"] = len(unique_scaffolds)

    out_df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")

# Example usage:
for i in range(1,15):
    process_csv(f"DATA/Step_1_Data/split_chunks/chunk_{i}.csv", 
                f"DATA/Step_2_Data/Ring_Features/ring_features_chunk{i}.csv")