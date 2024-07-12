import torch
from model import AntibodyQualityScorer
from utils.helpers import (
    sequence_to_tensor,
    structure_to_features,
    calculate_theoretical_perfection,
    experimental_analysis,
    calculate_pqs
)
import warnings
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

def predict():
    seq_input_size = 22
    struct_input_size = 4
    hidden_size = 256
    max_length = 100

    model = AntibodyQualityScorer(seq_input_size, struct_input_size, hidden_size)
    model.load_state_dict(torch.load('./models/trained_model.pth'))
    model.eval()

    new_sequence = "DIVMTQSPDSLAVSLGERATINCKSSQSLVDTSGNQITYLNWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLT"
    new_structure = "./data/test/pdb_files/3SDY.pdb"

    with torch.no_grad():
        sequence_tensor = sequence_to_tensor(new_sequence, max_length).unsqueeze(0).float()
        structure_tensor = structure_to_features(new_structure).unsqueeze(0).float()
        ml_prediction = model(sequence_tensor, structure_tensor).item()

        theoretical = calculate_theoretical_perfection(new_sequence)
        experimental = experimental_analysis(new_sequence)

        print(f"Theoretical: {theoretical}")
        print(f"ML Prediction: {ml_prediction}")
        print(f"Experimental: {experimental}")

        pqs = calculate_pqs(theoretical, ml_prediction, experimental)
        print(f"Predicted Protein Quality Score: {pqs:.2f}")

if __name__ == "__main__":
    predict()












