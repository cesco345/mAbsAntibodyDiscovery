import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
from antibody_dataset import AntibodyDataset
from model import AntibodyQualityScorer
from utils import (
    download_pdb,
    ensure_pdb_files,
    sequence_to_tensor,
    structure_to_features,
    calculate_theoretical_perfection,
    experimental_analysis,
    calculate_pqs
)

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for sequences, structures, exp_data in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences, structures)
            loss = criterion(outputs, exp_data)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sequences, structures, exp_data in test_loader:
            outputs = model(sequences, structures)
            loss = criterion(outputs, exp_data)
            total_loss += loss.item()
    average_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {average_loss:.4f}')

def load_test_data(csv_path, data_dir):
    df = pd.read_csv(csv_path)
    pdb_ids = df['PDB ID'].tolist()
    pdb_files = [os.path.join(data_dir, f"{pdb_id}.pdb") if pdb_id != "N/A" else None for pdb_id in pdb_ids]
    experimental_data = [0.0] * len(pdb_files)  # Replace with actual experimental data if available
    sequences = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS"] * len(pdb_files)  # Placeholder sequences
    return sequences, pdb_files, experimental_data

def main():
    # Training data (replace with actual data)
    train_sequences = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "DIQMTQSPSSLSASVGDRVTITC"]
    train_structures = ["./data/7L7D.pdb", "./data/7CR5.pdb"]
    train_experimental_data = [0.85, 0.92]

    # Load test data
    test_sequences, test_structures, test_experimental_data = load_test_data('./data/monoclonal_antibodies.csv', './data/test')

    # Ensure PDB files are available locally
    ensure_pdb_files(train_structures + [f for f in test_structures if f is not None])

    # Hyperparameters for the model
    seq_input_size = 22  # 20 amino acids + unknown + padding
    struct_input_size = 4  # Simplified structural features
    hidden_size = 64
    max_length = 100
    batch_size = 2
    learning_rate = 0.001
    epochs = 100

    # Prepare training dataset and dataloader
    train_tensor_sequences = [sequence_to_tensor(seq, max_length) for seq in train_sequences]
    train_tensor_structures = [structure_to_features(struct) for struct in train_structures]
    train_dataset = AntibodyDataset(train_tensor_sequences, train_tensor_structures, torch.tensor(train_experimental_data))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare test dataset and dataloader
    test_tensor_sequences = [sequence_to_tensor(seq, max_length) for seq in test_sequences]
    test_tensor_structures = [structure_to_features(struct) for struct in test_structures if struct is not None]
    test_dataset = AntibodyDataset(test_tensor_sequences, test_tensor_structures, torch.tensor(test_experimental_data))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AntibodyQualityScorer(seq_input_size, struct_input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, epochs)

    # Evaluate the model
    evaluate_model(model, test_loader, criterion)

    # Predict PQS for a new antibody
    model.eval()
    with torch.no_grad():
        new_sequence = sequence_to_tensor("EVQLVESGGGLVQPGGSLRLSCAASGFTFS", max_length).unsqueeze(0)
        new_structure = structure_to_features("./data/7L7D.pdb").unsqueeze(0)
        ml_prediction = model(new_sequence, new_structure)

        theoretical = calculate_theoretical_perfection("EVQLVESGGGLVQPGGSLRLSCAASGFTFS")
        experimental = experimental_analysis("EVQLVESGGGLVQPGGSLRLSCAASGFTFS")

        pqs = calculate_pqs(theoretical, ml_prediction, experimental)
        print(f"Predicted Protein Quality Score: {pqs:.2f}")

if __name__ == "__main__":
    main()



