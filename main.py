import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from antibody_dataset import AntibodyDataset
from model import AntibodyQualityScorer
from utils.helpers import (
    fetch_pdb_files_from_directory,
    sequence_to_tensor,
    structure_to_features,
    calculate_theoretical_perfection,
    experimental_analysis,
    calculate_pqs
)
from train import train_model, cross_validate_model, EarlyStopping
from sklearn.preprocessing import StandardScaler
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

def main():
    pdb_directory = './data/train/pdb_files'  # Update this to the correct PDB directory

    structures = fetch_pdb_files_from_directory(pdb_directory)
    if not structures:
        print("No PDB files available for training.")
        return

    sequences = ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS"] * len(structures)  # Placeholder sequences
    experimental_data = [0.85] * len(structures)  # Dummy experimental data

    num_pdb_files = len(structures)
    print(f"Number of PDB files used for training: {num_pdb_files}")

    if num_pdb_files < 3:
        print("Not enough samples for cross-validation.")
        return

    seq_input_size = 22
    struct_input_size = 4
    hidden_size = 256
    max_length = 100
    batch_size = 4
    learning_rate = 0.0001
    weight_decay = 0.01
    epochs = 100
    grad_clip = 1.0

    tensor_sequences = [sequence_to_tensor(seq, max_length).float() for seq in sequences]
    tensor_structures = [structure_to_features(struct).float() for struct in structures]

    scaler = StandardScaler()
    tensor_structures = [torch.tensor(scaler.fit_transform(struct.unsqueeze(0)).squeeze(0)).float() for struct in tensor_structures]

    dataset = AntibodyDataset(tensor_sequences, tensor_structures, torch.tensor(experimental_data).float())

    k_splits = min(5, num_pdb_files)
    avg_val_loss, avg_rmse, avg_mae = cross_validate_model(lambda: AntibodyQualityScorer(seq_input_size, struct_input_size, hidden_size),
                                        dataset, k=k_splits, epochs=epochs, batch_size=batch_size,
                                        learning_rate=learning_rate, grad_clip=grad_clip)

    print(f'Average Validation Loss: {avg_val_loss:.4f}, Average RMSE: {avg_rmse:.4f}, Average MAE: {avg_mae:.4f}')

    model = AntibodyQualityScorer(seq_input_size, struct_input_size, hidden_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    early_stopping = EarlyStopping(patience=20, delta=0.001)
    train_model(model, train_loader, criterion, optimizer, scheduler, epochs, grad_clip, early_stopping)

    # Save the trained model
    torch.save(model.state_dict(), './models/trained_model.pth')
    print("Model saved to './models/trained_model.pth'")

    model.eval()
    with torch.no_grad():
        new_sequence = sequence_to_tensor("EVQLVESGGGLVQPGGSLRLSCAASGFTFS", max_length).unsqueeze(0).float()
        new_structure = structure_to_features("./data/test/pdb_files/7L7D.pdb").unsqueeze(0).float()
        ml_prediction = model(new_sequence, new_structure)
        if isinstance(ml_prediction, torch.Tensor):
            ml_prediction = ml_prediction.item()

        theoretical = calculate_theoretical_perfection("EVQLVESGGGLVQPGGSLRLSCAASGFTFS")
        experimental = experimental_analysis("EVQLVESGGGLVQPGGSLRLSCAASGFTFS")

        print(f"Theoretical: {theoretical}")
        print(f"ML Prediction: {ml_prediction}")
        print(f"Experimental: {experimental}")

        pqs = calculate_pqs(theoretical, ml_prediction, experimental)
        print(f"Predicted Protein Quality Score: {pqs:.2f}")

if __name__ == "__main__":
    main()

































