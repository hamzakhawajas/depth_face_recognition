import os
import torch
import numpy as np
from siamese_network import SiameseNetwork
from torch.utils.data import DataLoader
from siamese_network import SiameseEmbeddingDataset


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")


net = SiameseNetwork()
state_dict = torch.load("siamese.pth", map_location=device)
net.load_state_dict(state_dict)

net.to(device)
net.eval()

numpy_load_path = 'new_entries/dataset_numpy_train2_padded'
embedding_save_path = 'new_entries/embeddings'

embedding_files = []
for person_folder in os.listdir(numpy_load_path):
    person_path = os.path.join(numpy_load_path, person_folder)
    if os.path.isdir(person_path):
        for file in os.listdir(person_path):
            if file.endswith('_data.npy'):
                embedding_files.append((os.path.join(person_path, file), person_folder))

embedding_dataset = SiameseEmbeddingDataset(numpy_load_path, embedding_files)
embedding_dataloader = DataLoader(embedding_dataset, shuffle=False)

embedding_dict = {}
for batch in embedding_dataloader:
    array, file_id = batch
    #print("fileid: ",file_id)
    array = array.view(1, 1, 300, 300).float().cuda() 
    with torch.no_grad():
        output = net.forward_one(array)
        output = output.cpu().numpy()
        
        # Manipulate file_id to get save path
        relative_path = os.path.relpath(file_id[0], numpy_load_path)
        save_path = os.path.join(embedding_save_path, relative_path)
        
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_path, output)
        print(save_path)
        