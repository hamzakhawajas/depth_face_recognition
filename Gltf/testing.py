import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2))

def extract_index(filename):
    # Assuming file format like '007_01_depth_data.npy', extract the '01' part
    parts = filename.split('_')
    return int(parts[1])  # This will convert '01' to 1

embeddings_folder_path = 'new_entries/embeddings'

embedding_files = []
for person_folder in os.listdir(embeddings_folder_path):
    person_path = os.path.join(embeddings_folder_path, person_folder)
    if os.path.isdir(person_path):
        files = sorted(os.listdir(person_path), key=lambda f: extract_index(f))
        for file in files:
            if file.endswith('_depth_data.npy'):
                index = extract_index(file)
                embedding_files.append((os.path.join(person_path, file), person_folder, index))

embeddings = [np.load(f[0]) for f in embedding_files]
labels = [f[1] for f in embedding_files]

embeddings_tensor = torch.tensor(np.stack(embeddings)).float().cuda()

test_dataset = TensorDataset(embeddings_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for batch_idx, (data,) in enumerate(test_dataloader):
    data = data.float().cuda()

    for i in range(data.shape[0]):
        for j in range(i + 1, data.shape[0]):
            output0, output1 = data[i].unsqueeze(0), data[j].unsqueeze(0)
            label = torch.tensor([1 if labels[i] == labels[j] else 0], dtype=torch.float32).cuda()
            
            distance = euclidean_distance(output0, output1).item()
            print(f"Person {labels[i]} (Embedding {embedding_files[i][2]})\tPerson {labels[j]} (Embedding {embedding_files[j][2]})\tEuclidean Distance: {distance:.4f}")

torch.cuda.empty_cache()
