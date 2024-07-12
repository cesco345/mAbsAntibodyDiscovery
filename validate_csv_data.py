import os
import torch
from torch.utils.data import Dataset, DataLoader

class PDBIDDataset(Dataset):
    def __init__(self, csv_path):
        self.data = []
        self.unique_pdb_ids = set()
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pdb_id = self.extract_pdb_id(line)
                    if pdb_id and pdb_id not in self.unique_pdb_ids:
                        self.data.append(pdb_id)
                        self.unique_pdb_ids.add(pdb_id)
                except Exception as e:
                    print(f"Error processing line: {line.strip()}. Error: {str(e)}")

    def extract_pdb_id(self, text):
        parts = text.split(',')
        for part in parts:
            cleaned = ''.join(c for c in part if c.isalnum())
            if len(cleaned) == 4:
                return cleaned
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pdb_id = self.data[idx]
        return torch.tensor([ord(char) for char in pdb_id], dtype=torch.float32)

class PDBIDClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 4)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_pdb_id_extractor(csv_path, output_path, num_epochs=50):
    dataset = PDBIDDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PDBIDClassifier()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

    with open(output_path, 'w') as f:
        for pdb_id in dataset.unique_pdb_ids:
            f.write(f"{pdb_id}\n")

    print(f"Total unique PDB IDs extracted: {len(dataset.unique_pdb_ids)}")

if __name__ == "__main__":
    csv_path = './data/monoclonal_antibodies.csv'
    output_path = './data/unique_pdb_ids.txt'
    train_pdb_id_extractor(csv_path, output_path)