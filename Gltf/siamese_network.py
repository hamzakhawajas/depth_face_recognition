import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

class SiameseNetwork(nn.Module): 
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(75264, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

    def forward_one(self, x):
        #print("beofre cnn", x.shape)
        x = self.cnn1(x)
        #print("after cnn", x.shape)
        x = x.view(x.size()[0], -1)
        #print("before fc cnn", x.shape)
        x = self.fc1(x)
        #print("after fc cnn", x.shape)
        #print("\n\n\n")
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    

class SiameseEmbeddingDataset(Dataset):
    def __init__(self, folder_path, files):
        self.folder_path = folder_path
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        #print(f"Debug: {self.files[index]}")
        file_path, folder_name = self.files[index]
        array = np.load(file_path)
        return torch.from_numpy(array), file_path

def extract_file_id(full_path):
    segments = full_path.split(os.sep)
    return os.sep.join(segments[-2:])

class SiameseTrainingDataset(Dataset):
    def __init__(self, data_folder, file_list):
        self.folder_path = data_folder
        self.file_list = file_list
        
        # Populate the list of available npy files
        for person_folder in os.listdir(data_folder):
            person_path = os.path.join(data_folder, person_folder)
            for file in os.listdir(person_path):
                file_path = os.path.join(person_path, file)
                self.file_list.append((file_path, person_folder))
        #print(self.file_list)
    
    def load_npy_file(self, path):
        return np.load(path)

    def __getitem__(self, index):
        file0_tuple = random.choice(self.file_list)
        
        # 50% chance to get same or different class
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            # Find another file from the same person
            while True:
                file1_tuple = random.choice(self.file_list)
                if file0_tuple[1] == file1_tuple[1]:
                    break
        else:
            # Find another file from a different person
            while True:
                file1_tuple = random.choice(self.file_list)
                if file0_tuple[1] != file1_tuple[1]:
                    break
        #print("Comparing {} with {} - Label: {}".format(file0_tuple[0], file1_tuple[0], int(file0_tuple[1] != file1_tuple[1])))

        # Load npy files
        data0 = self.load_npy_file(file0_tuple[0])
        data1 = self.load_npy_file(file1_tuple[0])
        
        data0 = np.expand_dims(data0, axis=0)
        data1 = np.expand_dims(data1, axis=0)

        file0_id = extract_file_id(file0_tuple[0])
        file1_id = extract_file_id(file1_tuple[0])

        file0_id = extract_file_id(file0_tuple[0])
        file1_id = extract_file_id(file1_tuple[0])
        

        #print("data0: ",data0, " data1: ", data1)
        # Return data and a label indicating whether the files are from the same class or not and also the file names
        return torch.from_numpy(data0), torch.from_numpy(data1), torch.from_numpy(np.array([int(file1_tuple[1] != file0_tuple[1])], dtype=np.float32)), file0_id, file1_id

    def __len__(self):
        return len(self.file_list)