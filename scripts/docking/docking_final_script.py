from pathlib import Path
import csv
import time
from vina import Vina

# ==========================
# CONFIGURATION
# ==========================

RECEPTOR = "data/docking_files/8XGK_clean.pdbqt"
LIGAND_DIR = "data/docking_files/top_0_1_pdbqt"
OUTPUT_DIR = "data/docking_files/output"
CSV_FILE = "data/docking_files/docking_scores.csv"

CENTER = [24.4234, 60.4669, 39.1378]
BOX_SIZE = [58.1984, 48.2553, 50.3424]

EXHAUSTIVENESS = 8
NUM_POSES = 9

# ==========================
# SETUP
# ==========================

start_time = time.time()

output_path = Path(OUTPUT_DIR)
output_path.mkdir(parents=True, exist_ok=True)

ligand_files = sorted(Path(LIGAND_DIR).glob("*.pdbqt"))
total_ligands = len(ligand_files)

v = Vina(sf_name="vina")
v.set_receptor(RECEPTOR)
v.compute_vina_maps(center=CENTER, box_size=BOX_SIZE)

# Create CSV with header if it does not exist
csv_path = Path(CSV_FILE)
if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ligand", "DockingScore_kcal_per_mol"])

# ==========================
# DOCKING LOOP
# ==========================

for idx, ligand_file in enumerate(ligand_files, start=1):
    ligand_start = time.time()

    try:
        print(f"\n[{idx}/{total_ligands}] Docking {ligand_file.name}")

        # Load ligand
        v.set_ligand_from_file(str(ligand_file))

        # Run docking
        v.dock(
            exhaustiveness=EXHAUSTIVENESS,
            n_poses=NUM_POSES
        )

        # Get best score
        best_score = float(v.energies()[0][0])

        # Save best pose
        output_file = output_path / f"{ligand_file.stem}_docked.pdbqt"
        v.write_pose(str(output_file), overwrite=True)

        # Immediately append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ligand_file.stem, best_score])

        print(f"Best score: {best_score:.3f} kcal/mol")

    except Exception as e:
        print(f"Failed: {ligand_file.name}")
        print(e)

        # Record failed docking immediately
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ligand_file.stem, None])

    ligand_time = time.time() - ligand_start
    total_time = time.time() - start_time

    avg_time = total_time / idx
    remaining = avg_time * (total_ligands - idx)

    print(f"Time for ligand : {ligand_time:.2f} s")
    print(f"Elapsed         : {total_time / 60:.2f} min")
    print(f"Estimated left  : {remaining / 60:.2f} min")

# ==========================
# SUMMARY
# ==========================

total_time = time.time() - start_time

hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = total_time % 60

print("\n===================================")
print("Docking completed.")
print(f"Total ligands processed : {total_ligands}")
print(f"Total runtime           : {hours}h {minutes}m {seconds:.2f}s")
print("===================================")