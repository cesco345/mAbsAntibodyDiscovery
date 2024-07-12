import os
import pandas as pd
from utils import download_pdb  # Ensure utils.py contains the download_pdb function

def fetch_pdb_files_from_csv(csv_path, output_dir):
    print(f"Reading CSV file from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"CSV file does not exist: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Check the total number of rows in the dataframe
    print(f"Total rows in CSV: {len(df)}")

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    downloaded_pdb_ids = set()
    skipped_pdb_ids = set()

    for idx, row in df.iterrows():
        pdb_id = str(row['PDB ID'])
        output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

        if os.path.exists(output_path):
            print(f"Skipping {pdb_id}, file already exists.")
            downloaded_pdb_ids.add(pdb_id)
            continue

        try:
            success = download_pdb(pdb_id, output_path)
            if success:
                print(f"Downloaded {pdb_id} to {output_path}")
                downloaded_pdb_ids.add(pdb_id)
            else:
                print(f"Skipping {pdb_id}, unable to download")
                skipped_pdb_ids.add(pdb_id)
        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            skipped_pdb_ids.add(pdb_id)

    print(f"Successfully downloaded {len(downloaded_pdb_ids)} PDB IDs.")
    print(f"Skipped {len(skipped_pdb_ids)} PDB IDs.")

    if skipped_pdb_ids:
        print(f"Skipped the following PDB IDs: {', '.join(skipped_pdb_ids)}")

if __name__ == "__main__":
    csv_path = './data/monoclonal_antibodies.csv'
    output_dir = './data/test'
    fetch_pdb_files_from_csv(csv_path, output_dir)















