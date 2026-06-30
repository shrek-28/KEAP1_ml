import os
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem

input_dir = "sdf_top_1_pct"
output_dir = "pdbqt_top_1_pct"

os.makedirs(output_dir, exist_ok=True)

preparator = MoleculePreparation()
failed = []

for file in os.listdir(input_dir):
    if not file.endswith(".sdf"):
        continue

    sdf_path = os.path.join(input_dir, file)
    pdbqt_path = os.path.join(output_dir, file.replace(".sdf", ".pdbqt"))

    try:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)

        if supplier is None or len(supplier) == 0:
            raise ValueError("No molecules found in SDF.")

        mol = supplier[0]

        if mol is None:
            raise ValueError("RDKit could not parse the molecule.")

        mol_setup = preparator.prepare(mol)[0]

        pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(mol_setup)

        if not success:
            raise RuntimeError(error_msg)

        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string)

        print(f"Converted {file}")

    except Exception as e:
        print(f"Failed: {file} -> {e}")
        failed.append(f"{file}\t{e}")

# Write failures to a text file
with open("failed_conversion.txt", "w") as f:
    f.write("Filename\tReason\n")
    f.write("=" * 80 + "\n")
    for entry in failed:
        f.write(entry + "\n")

print(f"\nSuccessfully converted: {len(os.listdir(output_dir))} files")
print(f"Failed: {len(failed)} files")
print("Failure log written to failed_conversion.txt")