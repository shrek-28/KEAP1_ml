import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors3D, rdMolDescriptors
from rdkit.Chem import GetPeriodicTable, rdFreeSASA, Descriptors3D
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from rdkit.Chem import rdMolTransforms

# Global periodic table object
PT = GetPeriodicTable()

# -----------------------------
# Property functions with logging
# -----------------------------
def log_and_calc(name, func, mol, mol_id):
    value = func(mol)
    print(f"{name} calculated for mol_{mol_id}")
    return value

def calc_radius_of_gyration(mol, mol_id): return log_and_calc("RadiusOfGyration", Descriptors3D.RadiusOfGyration, mol, mol_id)
def calc_asphericity(mol, mol_id): return log_and_calc("Asphericity", Descriptors3D.Asphericity, mol, mol_id)
def calc_eccentricity(mol, mol_id): return log_and_calc("Eccentricity", Descriptors3D.Eccentricity, mol, mol_id)
def calc_spherocity_index(mol, mol_id): return log_and_calc("SpherocityIndex", Descriptors3D.SpherocityIndex, mol, mol_id)
def calc_inertial_shape_factor(mol, mol_id): return log_and_calc("InertialShapeFactor", Descriptors3D.InertialShapeFactor, mol, mol_id)
def calc_pmi1(mol, mol_id): return log_and_calc("PMI1", Descriptors3D.PMI1, mol, mol_id)
def calc_pmi2(mol, mol_id): return log_and_calc("PMI2", Descriptors3D.PMI2, mol, mol_id)
def calc_pmi3(mol, mol_id): return log_and_calc("PMI3", Descriptors3D.PMI3, mol, mol_id)
def calc_pmi_ratio1(mol, mol_id): return log_and_calc("PMI_Ratio1", Descriptors3D.NPR1, mol, mol_id)
def calc_pmi_ratio2(mol, mol_id): return log_and_calc("PMI_Ratio2", Descriptors3D.NPR2, mol, mol_id)
def calc_surface_area(mol, mol_id): return log_and_calc("3D_SurfaceArea", Descriptors3D.SASA, mol, mol_id)

def calc_surface_area(mol, mol_id):
    # Use covalent radii as approximate SASA radii
    radii = [PT.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    print(f"3D_SurfaceArea calculated for mol_{mol_id}")
    return sasa

def calc_volume(mol, mol_id):
    radii = [PT.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    rg = Descriptors3D.RadiusOfGyration(mol)
    volume = (sasa * rg) / 3.0
    print(f"Volume (SASA approximation) calculated for mol_{mol_id}")
    return volume

def calc_bounding_box_volume(mol, mol_id):
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    vol = float(np.prod(maxs - mins))
    print(f"BoundingBoxVolume calculated for mol_{mol_id}")
    return vol

def calc_normalized_pmis(mol, mol_id):
    n1, n2 = Descriptors3D.NPR1(mol), Descriptors3D.NPR2(mol)
    print(f"Normalized PMIs calculated for mol_{mol_id}")
    return n1, n2

def calc_inertia_ratio(mol, mol_id):
    i1, i2, i3 = Descriptors3D.PMI1(mol), Descriptors3D.PMI2(mol), Descriptors3D.PMI3(mol)
    val = None if i3 == 0 else (i1 + i2) / i3
    print(f"InertiaRatio calculated for mol_{mol_id}")
    return val

def calc_covalent_radius_sum(mol, mol_id):
    val = sum(PT.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms())
    print(f"CovalentRadiusSum calculated for mol_{mol_id}")
    return val

def calc_dihedral_angle_range(mol, mol_id):
    conf = mol.GetConformer()
    dihedrals = []
    
    # Iterate over all torsions in the molecule
    for torsion in mol.GetSubstructMatches(Chem.MolFromSmarts('[!H]-[!H]-[!H]-[!H]')):
        i, j, k, l = torsion
        angle = rdMolTransforms.GetDihedralDeg(conf, i, j, k, l)
        dihedrals.append(angle)
    
    if dihedrals:
        rng = max(dihedrals) - min(dihedrals)
    else:
        rng = None  # None if no torsions exist
    
    print(f"DihedralAngleRange calculated for mol_{mol_id}")
    return rng

def calc_mean_conformer_energy(mol, mol_id):
    props = [float(conf.GetProp("Energy")) for conf in mol.GetConformers() if conf.HasProp("Energy")]
    val = np.mean(props) if props else None
    print(f"MeanConformerEnergy calculated for mol_{mol_id}")
    return val

def calc_lowest_conformer_energy(mol, mol_id):
    props = [float(conf.GetProp("Energy")) for conf in mol.GetConformers() if conf.HasProp("Energy")]
    val = np.min(props) if props else None
    print(f"LowestConformerEnergy calculated for mol_{mol_id}")
    return val

def calc_conformer_count(mol, mol_id):
    count = mol.GetNumConformers()
    print(f"ConformerCount calculated for mol_{mol_id}: {count}")
    return count

def calc_conformer_energies(mol, mol_id):
    confs = mol.GetConformers()
    energies = []

    for conf in confs:
        if conf.HasProp("_Energy"):
            energies.append(float(conf.GetProp("_Energy")))
        else:
            energies.append(0.0)  # fallback if energy not present

    if energies:
        mean_energy = sum(energies) / len(energies)
        lowest_energy = min(energies)
    else:
        mean_energy = 0.0
        lowest_energy = 0.0

    print(f"MeanConformerEnergy calculated for mol_{mol_id}: {mean_energy}")
    print(f"LowestConformerEnergy calculated for mol_{mol_id}: {lowest_energy}")
    return mean_energy, lowest_energy


def calc_shape_index(mol, mol_id):
    val = Descriptors3D.InertialShapeFactor(mol)
    print(f"ShapeIndex calculated for mol_{mol_id}")
    return val

def calc_apolar_surface_area(mol, mol_id):
    # Convert radii to tuple
    radii = tuple(PT.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms())

    # Create SASA options
    opts = rdFreeSASA.SASAOpts()
    opts.includeAtoms = True  # RDKit >= 2020.09

    # CalcSASA returns either float (total) or list (per atom) depending on RDKit version
    atom_sasa = rdFreeSASA.CalcSASA(mol, radii, opts=opts)

    # Ensure atom_sasa is iterable
    if isinstance(atom_sasa, float):
        # If float, per-atom SASA is unavailable → fallback: total SASA
        # We'll assign all SASA to each atom proportionally (approximation)
        atom_sasa = [atom_sasa / mol.GetNumAtoms()] * mol.GetNumAtoms()

    # Apolar atoms: C=6, S=16
    apolar_sasa = sum(sasa for i, sasa in enumerate(atom_sasa)
                      if mol.GetAtomWithIdx(i).GetAtomicNum() in (6, 16))

    print(f"APolarSurfaceArea calculated for mol_{mol_id}")
    return apolar_sasa

def extract_identifier_from_sdf(sdf_path):
    identifiers = []
    with open(sdf_path, 'r') as f:
        lines = f.readlines()

    current_id = None
    for i, line in enumerate(lines):
        if line.strip().startswith(">") and "IDENTIFIER" in line.upper():
            # The next non-empty line is the identifier
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                current_id = lines[j].strip()
                identifiers.append(current_id)
    return identifiers


def analyze_molecule(mol, mol_id):
    try:

        return {
            "Molecule_ID": mol_id,
            "RadiusOfGyration": calc_radius_of_gyration(mol, mol_id),
            "Asphericity": calc_asphericity(mol, mol_id),
            "Eccentricity": calc_eccentricity(mol, mol_id),
            "SpherocityIndex": calc_spherocity_index(mol, mol_id),
            "InertialShapeFactor": calc_inertial_shape_factor(mol, mol_id),
            "PMI1": calc_pmi1(mol, mol_id),
            "PMI2": calc_pmi2(mol, mol_id),
            "PMI3": calc_pmi3(mol, mol_id),
            "PMI_Ratio1": calc_pmi_ratio1(mol, mol_id),
            "PMI_Ratio2": calc_pmi_ratio2(mol, mol_id),
            "3D_Volume": calc_volume(mol, mol_id),
            "3D_SurfaceArea": calc_surface_area(mol, mol_id),
            "BoundingBoxVolume": calc_bounding_box_volume(mol, mol_id),
            "NormalizedPMI1": calc_normalized_pmis(mol, mol_id)[0],
            "NormalizedPMI2": calc_normalized_pmis(mol, mol_id)[1],
            "InertiaRatio": calc_inertia_ratio(mol, mol_id),
            "CovalentRadiusSum": calc_covalent_radius_sum(mol, mol_id),
            "DihedralAngleRange": calc_dihedral_angle_range(mol, mol_id),
            "ConformerCount": calc_conformer_count(mol, mol_id),
            "MeanConformerEnergy": calc_mean_conformer_energy(mol, mol_id),
            "LowestConformerEnergy": calc_lowest_conformer_energy(mol, mol_id),
            "ShapeIndex": calc_shape_index(mol, mol_id),
            "APolarSurfaceArea": calc_apolar_surface_area(mol, mol_id),
        }
    except Exception as e:
        return {"Molecule_ID": mol_id, "Error": str(e)}

from concurrent.futures import ProcessPoolExecutor, as_completed

def analyze_molecule_wrapper(args):
    mol, mol_id = args
    return analyze_molecule(mol, mol_id)

def analyze_sdf(sdf_path, output_csv="output.csv"):
    # Load molecules
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in suppl if mol is not None]

    # Extract identifiers from SDF text
    identifiers = extract_identifier_from_sdf(sdf_path)
    if len(identifiers) != len(mols):
        raise ValueError("Number of identifiers does not match number of molecules!")

    # Prepare arguments as tuples
    mol_args = list(zip(mols, identifiers))

    results = []
    cpu_count = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor safely
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = {executor.submit(analyze_molecule_wrapper, arg): arg[1] for arg in mol_args}
        for future in as_completed(futures):
            results.append(future.result())

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nAll properties calculated → Results saved to {output_csv}")
    return df

# -----------------------------
# Worker for one molecule
# -----------------------------
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # extra safe on Windows
    analyze_sdf("3D_sdf_tester_Files\\CNP00000161.sdf")