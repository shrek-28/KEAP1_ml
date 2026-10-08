# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Optional
from rdkit.Chem import GraphDescriptors
from mordred import Calculator, descriptors as mordred_descriptors



def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    G = nx.Graph()
    for a in mol.GetAtoms():
        G.add_node(a.GetIdx(), symbol=a.GetSymbol())
    for b in mol.GetBonds():
        G.add_edge(
            b.GetBeginAtomIdx(),
            b.GetEndAtomIdx(),
            order=float(b.GetBondTypeAsDouble()),
        )
    return G


def largest_connected_subgraph(G: nx.Graph) -> nx.Graph:
    if nx.is_connected(G):
        return G.copy()
    # handle empty graph safely
    if G.number_of_nodes() == 0:
        return nx.Graph()
    comp = max(nx.connected_components(G), key=len)
    return G.subgraph(comp).copy()


def exact_wiener_index(G: nx.Graph) -> float:
    if G.number_of_nodes() < 2:
        return 0.0
    H = largest_connected_subgraph(G)
    spl = dict(nx.all_pairs_shortest_path_length(H))
    total = 0
    nodes = list(H.nodes())
    for u, v in combinations(nodes, 2):
        total += spl[u][v]
    return float(total)


def exact_zagreb_index(G: nx.Graph) -> float:
    return float(sum(d ** 2 for _, d in G.degree()))


def exact_balaban_index(G: nx.Graph) -> float:
    H = largest_connected_subgraph(G)
    N = H.number_of_nodes()
    M = H.number_of_edges()
    if N < 2 or M == 0:
        return 0.0

    mu = M - N + 1
    spl = dict(nx.all_pairs_shortest_path_length(H))

    # w(u) = total distance from u to all others
    w = {u: sum(spl[u].values()) for u in H.nodes()}

    s = 0.0
    for u, v in H.edges():
        wu = w[u]
        wv = w[v]
        if wu <= 0 or wv <= 0:
            continue
        s += 1.0 / math.sqrt(wu * wv)

    return float(M / (mu + 1) * s)


def topo_radius(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    H = largest_connected_subgraph(G)
    try:
        return float(nx.radius(H))
    except nx.NetworkXError:
        return 0.0


def topo_diameter(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    H = largest_connected_subgraph(G)
    try:
        return float(nx.diameter(H))
    except nx.NetworkXError:
        return 0.0


def total_path_count(G: nx.Graph, max_len: int = 3) -> int:
    if G.number_of_nodes() < 2:
        return 0
    H = largest_connected_subgraph(G)
    nodes = list(H.nodes())
    count = 0
    for u, v in combinations(nodes, 2):
        for path in nx.all_simple_paths(H, u, v, cutoff=max_len):
            # only count paths with at least one edge
            if len(path) >= 2 and len(path) - 1 <= max_len:
                count += 1
    return int(count)


def path_complexity_index(G: nx.Graph, max_len: int = 3) -> float:
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return float(total_path_count(G, max_len=max_len) / n)


def distance_distribution_entropy(G: nx.Graph) -> float:
    H = largest_connected_subgraph(G)
    n = H.number_of_nodes()
    if n < 2:
        return 0.0
    spl = dict(nx.all_pairs_shortest_path_length(H))
    dists = []
    nodes = list(H.nodes())
    for u, v in combinations(nodes, 2):
        dists.append(spl[u][v])
    if not dists:
        return 0.0
    vals, counts = np.unique(dists, return_counts=True)
    probs = counts / counts.sum()
    # Shannon entropy base-2
    return float(-(probs * np.log2(probs)).sum())


def atom_information_content(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    symbols = [d["symbol"] for _, d in G.nodes(data=True)]
    vals, counts = np.unique(symbols, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def bond_information_content(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    orders = [d["order"] for _, _, d in G.edges(data=True)]
    vals, counts = np.unique(orders, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


# ----------------------------------------------------
# Mordred setup (exact, standardized implementations)
# ----------------------------------------------------
# We'll compute a *subset* from Mordred for the exact values where available,
# then fill the rest with our exact graph-based fallbacks.
MORDRED_KEEP = {
    # RDKit-like / classical exact descriptors from Mordred:
    "TopoPSA",          # Topological Polar Surface Area
    "BertzCT",          # Bertz complexity
    "WienerIndex",      # Wiener index
    "ZagrebIndex",      # Zagreb index
    "BalabanJ",         # Balaban J
    "Chi0", "Chi1", "Chi2", "Chi3", "Chi4",  # Chi indices (Mordred exposes these names)
    "KierFlex",         # Kier flexibility (Mordred's name)
}

_mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)


def mordred_subset(mol: Chem.Mol) -> dict:
    vals = _mordred_calc(mol).fill_missing(np.nan).asdict()
    out = {}
    for k in MORDRED_KEEP:
        out[k] = vals.get(k, np.nan)
    # ensure primitive types (for JSON/CSV safety)
    for k, v in out.items():
        if isinstance(v, (np.generic,)):
            out[k] = np.asscalar(v) if hasattr(np, "asscalar") else v.item()
    return out


def rdkit_core_descriptors(mol: Chem.Mol) -> dict:
    return {
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),           
        "BertzCT_RDKit": float(Descriptors.BertzCT(mol)),        
    }

def safe_kier_flex(mol):
    try:
        val = rdMolDescriptors.CalcKierFlexibility(mol)
        if np.isnan(val):
            return 0.0   # or None, depending on your needs
        return val
    except Exception:
        return 0.0


def compute_all_descriptors_for_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"_parse_ok": False}

    # Graph build
    G = mol_to_nx(mol)

    # Mordred exact set (preferred where available)
    md = mordred_subset(mol)

    # RDKit exacts
    rd = rdkit_core_descriptors(mol)

    # Graph-exact fallbacks (used if Mordred value is NaN or to add extras not in Mordred)
    wiener = exact_wiener_index(G) if np.isnan(md.get("WienerIndex", np.nan)) else md["WienerIndex"]
    zagreb = exact_zagreb_index(G) if np.isnan(md.get("ZagrebIndex", np.nan)) else md["ZagrebIndex"]
    balaban = exact_balaban_index(G) if np.isnan(md.get("BalabanJ", np.nan)) else md["BalabanJ"]

    # Additional required metrics (exact)
    top_diam = topo_diameter(G)
    top_rad = topo_radius(G)
    total_paths = total_path_count(G, max_len=3)            # exact up to length 3
    path_c_idx = path_complexity_index(G, max_len=3)
    dist_entropy = distance_distribution_entropy(G)
    atom_ic = atom_information_content(G)
    bond_ic = bond_information_content(G)

    out = {
        "TPSA": rd["TPSA"],

        "WienerIndex": float(wiener),
        "ZagrebIndex": float(zagreb),
        "BalabanIndex": float(balaban),

        "BertzCTIndex": float(md.get("BertzCT", np.nan))
            if not np.isnan(md.get("BertzCT", np.nan)) else rd["BertzCT_RDKit"],

        "Chi0": GraphDescriptors.Chi0(mol),
        "Chi1": GraphDescriptors.Chi1(mol),
        "Chi2": GraphDescriptors.Chi2n(mol),
        "Chi3": GraphDescriptors.Chi3n(mol),
        "Chi4": GraphDescriptors.Chi4n(mol),

        "KierFlexibilityIndex": safe_kier_flex(mol),

        "AtomInformationContent": float(atom_ic),
        "BondInformationContent": float(bond_ic),

        "TopologicalDiameter": float(top_diam),
        "TopologicalRadius": float(top_rad),

        "TotalPathCount": int(total_paths),
        "FragmentComplexityScore": int(nx.number_connected_components(G)),   # exact (components count)
        "PathComplexityIndex": float(path_c_idx),

        "GraphDistanceDistributionEntropy": float(dist_entropy),

        "_parse_ok": True,
    }
    return out


def compute_descriptors_from_csv(
    input_csv: str,
    id_col: str = "identifier",
    smiles_col: str = "canonical_smiles",
    output_csv: Optional[str] = None,
    log_interval: int = 500,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    results = []

    for idx, row in df.iterrows():
        mol_id = row[id_col]
        smi = row[smiles_col]

        desc = compute_all_descriptors_for_smiles(smi)
        desc[id_col] = mol_id
        desc[smiles_col] = smi
        results.append(desc)

        if (idx + 1) % log_interval == 0:
            print(f"Processed {idx + 1} molecules...")

    out_df = pd.DataFrame(results)

    # Keep only successfully parsed rows
    out_df = out_df[out_df["_parse_ok"] == True].drop(columns=["_parse_ok"])

    # Reorder columns: id, smiles, then descriptors
    cols = [id_col, smiles_col] + [c for c in out_df.columns if c not in (id_col, smiles_col)]
    out_df = out_df[cols]

    if output_csv:
        out_df.to_csv(output_csv, index=False)
        print(f"Saved descriptors to {output_csv}")

    return out_df

if __name__ == "__main__":
    df = compute_descriptors_from_csv(
        "C:/Users/ASUS/Downloads/Project/chunks/chunk_14.csv",
        id_col="identifier",
        smiles_col="canonical_smiles",
        output_csv="molecular_descriptors_chunk14.csv",
        log_interval=10,
    )
    print(df.head())