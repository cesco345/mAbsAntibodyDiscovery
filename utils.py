from Bio.PDB import PDBParser
import torch
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
import requests
import os
import json

warnings.simplefilter('ignore')

def download_pdb(pdb_id, save_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except requests.RequestException as e:
        print(f"Error downloading {pdb_id}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)  # Remove partial file if download failed
        return False

def ensure_pdb_files(pdb_files):
    for pdb_file in pdb_files:
        if pdb_file and not os.path.exists(pdb_file):
            pdb_id = os.path.basename(pdb_file).replace('.pdb', '')
            download_pdb(pdb_id, pdb_file)

def sequence_to_tensor(sequence, max_length):
    aa_to_int = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    int_sequence = [aa_to_int.get(aa, 20) for aa in sequence]
    int_sequence = int_sequence[:max_length] + [21] * (max_length - len(int_sequence))
    tensor = torch.zeros(max_length, 22)
    tensor[range(max_length), int_sequence] = 1
    return tensor

def structure_to_features(structure):
    parser = PDBParser()
    structure = parser.get_structure("antibody", structure)
    feature_vector = [
        len(structure),  # Number of models
        len(list(structure.get_chains())),  # Number of chains
        len(list(structure.get_residues())),  # Number of residues
        len(list(structure.get_atoms()))  # Number of atoms
    ]
    return torch.tensor(feature_vector, dtype=torch.float32)

def calculate_theoretical_perfection(sequence):
    analysis = ProteinAnalysis(sequence)
    return {
        "molecular_weight": analysis.molecular_weight(),
        "isoelectric_point": analysis.isoelectric_point(),
        "aromaticity": analysis.aromaticity(),
        "instability_index": analysis.instability_index(),
        "gravy": analysis.gravy(),
    }

def experimental_analysis(antibody):
    return {
        "sds_page_band": np.random.uniform(0.9, 1.1),
        "sec_elution_volume": np.random.uniform(0.9, 1.1),
        "maldi_tof_mass": np.random.uniform(0.9, 1.1),
        "dls_radius": np.random.uniform(0.9, 1.1),
        "thermal_melting_temp": np.random.uniform(60, 80),
        "aggregation_temp": np.random.uniform(50, 70),
    }


def calculate_pqs(theoretical, ml_prediction, experimental):
    theoretical_score = sum(theoretical.values()) / len(theoretical)
    ml_score = ml_prediction
    exp_score = sum(experimental.values()) / len(experimental)

    # Normalize each score to be within a reasonable range, e.g., 0-1
    theoretical_score /= max(theoretical.values())
    ml_score = (ml_score - min(ml_prediction, 0)) / (max(ml_prediction, 1) - min(ml_prediction, 0))
    exp_score /= max(experimental.values())

    # Weight each component
    pqs = (theoretical_score * 0.2 + ml_score * 0.3 + exp_score * 0.5) * 100
    return min(max(pqs, 1), 100)

def fetch_pdb_files_from_directory(directory='./data'):

    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return []

    pdb_ids = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.pdb')]
    if not pdb_ids:
        print(f"No PDB files found in directory '{directory}'.")
        return []

    structure_paths = [os.path.join(directory, f"{pdb_id}.pdb") for pdb_id in pdb_ids]
    ensure_pdb_files(structure_paths)
    return structure_paths



