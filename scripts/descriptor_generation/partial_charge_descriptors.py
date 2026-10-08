import pandas as pd
import torch
from qmdesc import ReactivityDescriptorHandler
import argparse
from tqdm import tqdm  

# Allow argparse.Namespace in PyTorch safe globals
torch.serialization.add_safe_globals([argparse.Namespace])

# Initialize QMDesc handler
handler = ReactivityDescriptorHandler()

# Load CSV
df = pd.read_csv("DATA/Step_1_Data/split_chunks/chunk_1.csv")  

partial_charge_max_list = []
partial_charge_min_list = []
total_partial_charge_list = []

# Batch processing
batch_size = 500
smiles_list = df["canonical_smiles"].tolist()
processed_count = 0  # Counter for logging

for start in tqdm(range(0, len(smiles_list), batch_size)):
    batch_smiles = smiles_list[start:start + batch_size]
    for smiles in batch_smiles:
        try:
            descriptors = handler.predict(smiles)
            charges = descriptors["partial_charge"]
            
            partial_charge_max_list.append(max(charges))
            partial_charge_min_list.append(min(charges))
            total_partial_charge_list.append(sum(charges))
        except Exception as e:
            partial_charge_max_list.append(None)
            partial_charge_min_list.append(None)
            total_partial_charge_list.append(None)
            print(f"Failed for SMILES: {smiles}, error: {e}")
        
        # Increment counter and print every 20 molecules
        processed_count += 1
        if processed_count % 5 == 0:
            print(f"{processed_count} molecules processed...")

# Append results as new columns
df["partial_charge_max"] = partial_charge_max_list
df["partial_charge_min"] = partial_charge_min_list
df["total_partial_charge"] = total_partial_charge_list

# Save output CSV
df.to_csv("molecules_with_charges.csv", index=False)
print("Done! Output saved to molecules_with_charges.csv")