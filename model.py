import torch
import torch.nn as nn

class AntibodyQualityScorer(nn.Module):
    def __init__(self, seq_input_size, struct_input_size, hidden_size):
        super(AntibodyQualityScorer, self).__init__()
        self.seq_lstm = nn.LSTM(seq_input_size, hidden_size, batch_first=True)
        self.struct_fc = nn.Linear(struct_input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5)  # Add dropout layer
        self.combine_fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, seq, struct):
        _, (seq_hidden, _) = self.seq_lstm(seq)  # Process sequence data with LSTM
        struct_features = self.struct_fc(struct)  # Process structural data with a fully connected layer
        combined = torch.cat((seq_hidden[-1], struct_features), dim=1)  # Combine both features
        combined = self.dropout(combined)  # Apply dropout
        output = self.combine_fc(combined)  # Final prediction
        return output.squeeze()

