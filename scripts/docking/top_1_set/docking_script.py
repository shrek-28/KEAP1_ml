import os
import subprocess
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

vina_exe = "vina"
config_file = "scripts/docking/representatives/vina_config.txt"

ligand_dir = "pdbqt_top_1_pct"          # Folder containing all PDBQT files
results_dir = "data/docked_top_1_pct_pdbqt"
csv_path = "data/docked_top_1_pct_pdbqt/docking_results.csv"

MAX_WORKERS = max(1, os.cpu_count() - 8)

os.makedirs(results_dir, exist_ok=True)


def dock_ligand(ligand):
    ligand_path = os.path.join(ligand_dir, ligand)

    output_file = os.path.join(
        results_dir,
        ligand.replace(".pdbqt", "_out.pdbqt")
    )

    if os.path.exists(output_file):
        return ligand, "SKIPPED"

    cmd = [
        vina_exe,
        "--config", config_file,
        "--ligand", ligand_path,
        "--out", output_file,
        "--cpu", "1"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        score = "NA"
        for line in result.stdout.splitlines():
            if line.strip().startswith("1 "):
                score = line.split()[1]
                break

        return ligand, score

    except subprocess.CalledProcessError:
        return ligand, "FAILED"


# -------------------------------------------------------------------
# Resume support
# -------------------------------------------------------------------

processed = set()

if os.path.exists(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                processed.add(row[0])

ligands = [
    f for f in os.listdir(ligand_dir)
    if f.endswith(".pdbqt") and f not in processed
]

print(f"Total remaining ligands: {len(ligands)}")

results_buffer = []

with ProcessPoolExecutor(MAX_WORKERS) as executor:

    futures = {
        executor.submit(dock_ligand, ligand): ligand
        for ligand in ligands
    }

    for future in tqdm(as_completed(futures), total=len(futures)):

        ligand, score = future.result()
        results_buffer.append([ligand, score])

        if len(results_buffer) >= 50:
            write_header = not os.path.exists(csv_path)

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)

                if write_header:
                    writer.writerow(["Ligand", "Score"])

                writer.writerows(results_buffer)

            results_buffer = []

# Final flush
if results_buffer:
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["Ligand", "Score"])

        writer.writerows(results_buffer)

print("All docking completed.")