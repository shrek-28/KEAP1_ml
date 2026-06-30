from pathlib import Path
import csv
import time
from vina import Vina

# =====================================================
# CONFIGURATION
# =====================================================

RECEPTOR = "data/docking_files/8XGK_clean.pdbqt"
LIGAND_DIR = "data/docking_files/pdbqt_top_1_pct"
OUTPUT_DIR = "data/docking_files/output"
CSV_FILE = "data/docking_files/docking_scores.csv"

CENTER = [24.4234, 60.4669, 39.1378]
BOX_SIZE = [58.1984, 48.2553, 50.3424]

EXHAUSTIVENESS = 8
NUM_POSES = 9

# =====================================================
# INITIALIZATION
# =====================================================

start_time = time.time()

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

ligand_files = sorted(Path(LIGAND_DIR).glob("*.pdbqt"))
total_ligands = len(ligand_files)

print(f"Found {total_ligands} ligands.")

vina = Vina(sf_name="vina")
vina.set_receptor(RECEPTOR)

print("Computing affinity maps...")
vina.compute_vina_maps(
    center=CENTER,
    box_size=BOX_SIZE
)

csv_path = Path(CSV_FILE)

if not csv_path.exists():
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ligand",
            "BestScore_kcal_per_mol"
        ])

# =====================================================
# DOCKING
# =====================================================

for idx, ligand in enumerate(ligand_files, start=1):

    ligand_start = time.time()

    print("\n==================================================")
    print(f"[{idx}/{total_ligands}] {ligand.name}")

    try:

        vina.set_ligand_from_file(str(ligand))

        vina.dock(
            exhaustiveness=EXHAUSTIVENESS,
            n_poses=NUM_POSES
        )

        energies = vina.energies()

        if len(energies) == 0:
            raise RuntimeError("Docking returned zero poses.")

        best_score = float(energies[0][0])

        output_file = output_dir / f"{ligand.stem}_docked.pdbqt"

        # Save ALL docked poses (up to NUM_POSES)
        vina.write_poses(
            str(output_file),
            n_poses=NUM_POSES,
            overwrite=True
        )

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ligand.stem,
                best_score
            ])

        print(f"Best score : {best_score:.3f} kcal/mol")
        print(f"Saved      : {output_file.resolve()}")
        print(f"Poses      : {len(energies)}")

    except Exception as e:

        print("Docking failed.")
        print(e)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                ligand.stem,
                None
            ])

    ligand_time = time.time() - ligand_start
    elapsed = time.time() - start_time

    avg_time = elapsed / idx
    remaining = avg_time * (total_ligands - idx)

    print(f"Time for ligand : {ligand_time:.2f} s")
    print(f"Elapsed         : {elapsed/60:.2f} min")
    print(f"Remaining       : {remaining/60:.2f} min")

# =====================================================
# SUMMARY
# =====================================================

total_runtime = time.time() - start_time

hours = int(total_runtime // 3600)
minutes = int((total_runtime % 3600) // 60)
seconds = total_runtime % 60

print("\n==================================================")
print("Docking completed.")
print(f"Ligands processed : {total_ligands}")
print(f"Runtime           : {hours}h {minutes}m {seconds:.2f}s")
print("==================================================")