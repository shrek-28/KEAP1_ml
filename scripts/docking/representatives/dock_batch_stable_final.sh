#!/bin/bash

mkdir -p cluster_csvs

echo "Detecting ligand folders..."

folders=(PDBQT_Molecules/*)
folders=("${folders[@]##*/}")

to_run=()

for folder in "${folders[@]}"; do
    if [ -f "cluster_csvs/${folder}.csv" ]; then
        echo "Skipping $folder (already has CSV)"
    else
        to_run+=("$folder")
    fi
done

if [ ${#to_run[@]} -eq 0 ]; then
    echo "Nothing to run. All folders already processed."
    exit 0
fi

echo ""
echo "Starting docking for:"
echo "${to_run[@]}"
echo ""

python3 dock_cluster.py "${to_run[@]}"
