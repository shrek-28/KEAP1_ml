import os
import subprocess
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

vina_exe = "vina"
config_file = "vina_config.txt"

ligand_root = "PDBQT_Molecules"
results_dir = "cluster_results"
csv_dir = "cluster_csvs"

MAX_WORKERS = max(1, os.cpu_count() - 8)

os.makedirs(results_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python dock_cluster.py folder1 folder2 ...")
    sys.exit()

folders = sys.argv[1:]


def dock_ligand(args):
    folder, ligand = args

    folder_path = os.path.join(ligand_root, folder)
    ligand_path = os.path.join(folder_path, ligand)

    output_file = os.path.join(
        results_dir,
        f"{folder}_{ligand.replace('.pdbqt','_out.pdbqt')}"
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

        for line in result.stdout.splitlines():
            if line.strip().startswith("1 "):
                score = line.split()[1]
                return ligand, score

        return ligand, "NA"

    except subprocess.CalledProcessError as e:
        return ligand, f"FAILED"


for folder in folders:

    folder_path = os.path.join(ligand_root, folder)

    if not os.path.isdir(folder_path):
        print("Invalid folder:", folder)
        continue

    print(f"\n===== Processing {folder} =====")

    csv_path = os.path.join(csv_dir, f"{folder}.csv")

    ligands = [l for l in os.listdir(folder_path) if l.endswith(".pdbqt")]

    # Load already processed ligands (resume support)
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])

    ligands = [l for l in ligands if l not in processed]

    print(f"Total ligands: {len(ligands)} | Skipped: {len(processed)}")

    if not ligands:
        continue

    results_buffer = []

    with ProcessPoolExecutor(MAX_WORKERS) as executor:

        futures = {
            executor.submit(dock_ligand, (folder, ligand)): ligand
            for ligand in ligands
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=folder):

            ligand, score = future.result()
            results_buffer.append([ligand, score])

            # Write in batches (every 50)
            if len(results_buffer) >= 50:
                with open(csv_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    if os.stat(csv_path).st_size == 0:
                        writer.writerow(["Ligand", "Score"])
                    writer.writerows(results_buffer)
                results_buffer = []

    # Final flush
    if results_buffer:
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if os.stat(csv_path).st_size == 0:
                writer.writerow(["Ligand", "Score"])
            writer.writerows(results_buffer)

print("\nAll docking completed.")
