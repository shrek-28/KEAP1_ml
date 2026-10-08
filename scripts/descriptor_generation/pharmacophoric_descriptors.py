from rdkit import Chem
import pandas as pd 

smarts_patterns = {
    "Hydroxyl": "[OX2H]",                          
    "Carbonyl": "[CX3]=[OX1]",                     
    "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",            
    "Nitro": "[NX3](=O)=O",                        
    "Halogen": "[F,Cl,Br,I]",                      
    "Sulfonyl": "S(=O)(=O)",                       
    "Ester": "[CX3](=O)[OX2H0][#6]",               
    "Ether": "[OD2]([#6])[#6]",                    
    "Thiol": "[#16X2H]",                           
    "Phenol": "c[OH]",                             
    "Carboxylic_Acid": "C(=O)[OH]",                
    "Imine": "[CX2]=[NX2]",                        
    "Alkyne": "[CX2]#C",                           
    "Nitrile": "[CX2]#N",                          
    "Aldehyde": "[CX3H1](=O)[#6]",                 
    "Ketone": "[#6][CX3](=O)[#6]",                 
}

def lactone_count(mol):
    patt = Chem.MolFromSmarts("C(=O)O")
    return sum(1 for match in mol.GetSubstructMatches(patt) if mol.GetAtomWithIdx(match[0]).IsInRing())

def lactam_count(mol):
    patt = Chem.MolFromSmarts("C(=O)N")
    return sum(1 for match in mol.GetSubstructMatches(patt) if mol.GetAtomWithIdx(match[0]).IsInRing())

def halogenated_aromatics_count(mol):
    patt = Chem.MolFromSmarts("c[F,Cl,Br,I]")
    return len(mol.GetSubstructMatches(patt))

def quaternary_carbons_count(mol):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetDegree() == 4)

def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: None for k in list(smarts_patterns.keys()) + ["Lactone","Lactam","Halogenated_Aromatics","Quaternary_Carbons"]}

    results = {}
    for name, smarts in smarts_patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        results[name] = len(mol.GetSubstructMatches(patt))

    # special features
    results["Lactone"] = lactone_count(mol)
    results["Lactam"] = lactam_count(mol)
    results["Halogenated_Aromatics"] = halogenated_aromatics_count(mol)
    results["Quaternary_Carbons"] = quaternary_carbons_count(mol)

    return results

# --- main pipeline ---
def process_csv(input_csv, smiles_col="canonical_smiles", output_csv="descriptors_output.csv"):
    df = pd.read_csv(input_csv)

    all_results = []
    for i, smi in enumerate(df[smiles_col]):
        res = calculate_descriptors(smi)
        all_results.append(res)

        # # log every 100 molecules
        # if (i + 1) % 1 == 0:
        #     print(f"[INFO] Processed {i+1} molecules...")

    results_df = pd.DataFrame(all_results)
    final_df = pd.concat([df, results_df], axis=1)
    final_df.to_csv(output_csv, index=False)
    print(f"[INFO] Finished processing. Output saved to {output_csv}")
    return final_df

# --- run it ---
if __name__ == "__main__":
    for i in range(1, 15):
        output = process_csv(
            f"DATA/Step_1_Data/split_chunks/chunk_{i}.csv",
            smiles_col="canonical_smiles",
            output_csv=f"DATA/Step_2_Data/Pharmacophoric_Features/pharmacophoric_chunk{i}.csv"
        )