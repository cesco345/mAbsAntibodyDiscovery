import os
import torch
from model import AntibodyQualityScorer
from utils.helpers import (
    sequence_to_tensor,
    structure_to_features,
    calculate_theoretical_perfection,
    experimental_analysis,
    calculate_pqs
)
import joblib


def load_model(model_path, seq_input_size, struct_input_size, hidden_size):
    model = AntibodyQualityScorer(seq_input_size, struct_input_size, hidden_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_quality(sequence, pdb_file, model, scaler, max_length):
    sequence_tensor = sequence_to_tensor(sequence, max_length).unsqueeze(0).float()
    structure_tensor = structure_to_features(pdb_file).unsqueeze(0).float()
    structure_tensor = torch.tensor(scaler.transform(structure_tensor), dtype=torch.float32)

    with torch.no_grad():
        ml_prediction = model(sequence_tensor, structure_tensor).item()

    theoretical = calculate_theoretical_perfection(sequence)
    experimental = experimental_analysis(sequence)

    pqs = calculate_pqs(theoretical, ml_prediction, experimental)

    return theoretical, ml_prediction, experimental, pqs


def main():
    seq_input_size = 22
    struct_input_size = 4
    hidden_size = 256
    max_length = 100
    model_path = './models/trained_model.pth'
    scaler_path = 'scaler.pkl'

    # Load model and scaler
    model = load_model(model_path, seq_input_size, struct_input_size, hidden_size)
    scaler = joblib.load(scaler_path)

    # List of new proteins (sequences and corresponding pdb files)
    proteins = [
        {"sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFS", "pdb_file": "./data/test/pdb_files/7L7D.pdb"},
        # Add more protein sequences and pdb files here
    ]

    for protein in proteins:
        theoretical, ml_prediction, experimental, pqs = predict_quality(protein["sequence"], protein["pdb_file"], model,
                                                                        scaler, max_length)

        print(f"Protein: {protein['sequence']}")
        print(f"Theoretical: {theoretical}")
        print(f"ML Prediction: {ml_prediction}")
        print(f"Experimental: {experimental}")
        print(f"Predicted Protein Quality Score: {pqs:.2f}")
        print()


if __name__ == "__main__":
    main()
