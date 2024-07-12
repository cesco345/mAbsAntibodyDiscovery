import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import download_pdb, ensure_pdb_files


def fetch_pdb_files_from_directory(directory='./data/test/'):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    pdb_ids = [f.split('.')[0] for f in os.listdir(directory) if f.endswith('.pdb')]
    if not pdb_ids:
        print(f"No PDB files found in directory '{directory}'.")
        return

    structure_paths = [os.path.join(directory, f"{pdb_id}.pdb") for pdb_id in pdb_ids]
    ensure_pdb_files(structure_paths)

    print(f"Fetched {len(pdb_ids)} PDB files from directory '{directory}'.")


if __name__ == "__main__":
    pdb_directory = './data/test/'  # Specify the directory containing PDB files

    fetch_pdb_files_from_directory(pdb_directory)




