import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, MolSurf
from rdkit.Chem.QED import properties

def calc_LogS_ESOL(mol):
    MW = Descriptors.MolWt(mol)
    LogP = Crippen.MolLogP(mol)
    RotB = Lipinski.NumRotatableBonds(mol)
    AP = rdMolDescriptors.CalcNumAromaticRings(mol) / max(1, rdMolDescriptors.CalcNumRings(mol))
    return 0.16 - 0.63*LogP - 0.0062*MW + 0.066*RotB - 0.74*AP

def calc_Apolar_SA(mol):
    contribs = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    apolar = 0.0
    for atom, area in zip(mol.GetAtoms(), contribs):
        if atom.GetAtomicNum() not in (7,8,15,16):  # exclude polar atoms (N,O,P,S)
            apolar += area
    return apolar

def calc_descriptors(smiles_list):
    data = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        desc = {}
        desc["SMILES"] = smi
        desc["MolecularWeight"] = Descriptors.MolWt(mol)
        desc["HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
        desc["TotalAtomCount"] = mol.GetNumAtoms()
        desc["RotatableBonds"] = Lipinski.NumRotatableBonds(mol)
        desc["RingCount"] = rdMolDescriptors.CalcNumRings(mol)
        desc["AromaticRingCount"] = rdMolDescriptors.CalcNumAromaticRings(mol)
        desc["FractionSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
        desc["HBDonors"] = Lipinski.NumHDonors(mol)
        desc["HBAcceptors"] = Lipinski.NumHAcceptors(mol)
        desc["TPSA"] = MolSurf.TPSA(mol)
        desc["ApolarSurfaceArea"] = calc_Apolar_SA(mol)
        desc["FormalCharge"] = Chem.GetFormalCharge(mol)
        desc["LogP"] = Crippen.MolLogP(mol)
        desc["LogS_ESOL"] = calc_LogS_ESOL(mol)
        desc["MolarRefractivity"] = Crippen.MolMR(mol)
        desc["HeteroatomCount"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
        desc["StereocenterCount"] = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        desc["DoubleBondCount"] = sum(1 for b in mol.GetBonds() if b.GetBondType()==Chem.rdchem.BondType.DOUBLE)
        desc["TripleBondCount"] = sum(1 for b in mol.GetBonds() if b.GetBondType()==Chem.rdchem.BondType.TRIPLE)
        desc["BridgeheadAtomCount"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        data.append(desc)
    return pd.DataFrame(data)