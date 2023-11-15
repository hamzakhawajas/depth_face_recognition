import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from siamese_network import SiameseNetwork
from siamese_network import SiameseTrainingDataset
from siamese_network import ContrastiveLoss


folder_path = 'new_entries/dataset_numpy_train2_padded'
num_epochs = 15

all_files = []
for person_folder in os.listdir(folder_path):
    person_path = os.path.join(folder_path, person_folder)
    if os.path.isdir(person_path):
        for file in os.listdir(person_path):
            if file.endswith('_data.npy'):
                all_files.append((os.path.join(person_path, file), person_folder))

train_dataset = SiameseTrainingDataset(folder_path, all_files)
train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=8, batch_size=64)
#print("all files: ",all_files)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss().cuda()
optimizer = optim.Adam(net.parameters(), lr=0.00001)



error_array = []
valid_error_array = []
for epoch in range(num_epochs):
    for i, (array1, array2, label, file0, file1) in enumerate(train_dataloader, 0):
        array1, array2, label = array1.float().cuda(), array2.float().cuda(), label.float().cuda()
        
        optimizer.zero_grad()
        output1, output2 = net(array1, array2)
        loss = criterion(output1, output2, label)
        
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
        
    error_array.append(loss.item())

model_name="siamese.pth"
torch.save(net.state_dict(), model_name)
print("Model Saved: " + str(model_name))
net.eval()

