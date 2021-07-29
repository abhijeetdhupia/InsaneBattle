# Define a custom model which takes 34 data points and output 8 classes 

import numpy as np
import pandas as pd
from natsort import natsorted
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# set seed 
np.random.seed(42)
torch.manual_seed(42)

EPOCHS = 50
LEARNING_RATE = 0.001
DROPOUT = 0.5

# Load data
test = pd.read_csv("./data/test_data.csv")
data = pd.read_csv("./data/train_data.csv")
labels = pd.read_csv("./data/train_labels.csv")
train_data_df = data.drop(['id'], axis=1)
train_labels = labels.drop(['id'], axis=1)
test_data = test.drop(['id'], axis=1)

# Split the data into training and validation data (80%-20%)
train_data, val_data, train_labels, val_labels = train_test_split(train_data_df, train_labels, test_size=0.2)

# Convert the data into tensors
train_data = torch.from_numpy(train_data.values).float()
val_data = torch.from_numpy(val_data.values).float()
train_labels = torch.from_numpy(train_labels.values).long()
val_labels = torch.from_numpy(val_labels.values).long()
test_data = torch.from_numpy(test_data.values).float()

# model with dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(34, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 8)
        self.dropout = nn.Dropout(p=DROPOUT)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x)) 
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = F.softmax(self.fc8(x), dim=1)
        return x

# Create model
# model = Net()
model = Net() 

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_labels.squeeze())
    loss.backward()
    optimizer.step()
    print("Epoch: {}, Training Loss: {}".format(epoch, loss.item()))

# Test the model on validation set and plot the prediction  

with torch.no_grad():
    test_output = model(val_data)
    _, predicted = torch.max(test_output.data, 1)
    # print(predicted)
    # print(val_labels)
    # print(test_output)
    correct = 0
    total = 0
    for i in range(0, val_labels.size()[0]):
        total += 1
        if predicted[i] == val_labels.data[i]:
            correct += 1
    print("Accuracy: {} %".format(100 * correct / total))

    # Plot the confusion matrix
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7']

    # Compute and plot the confusion matrix
    cm = confusion_matrix(val_labels.numpy(), predicted.numpy())
    np.set_printoptions(precision=2)
    print(cm)
 

# Save the model
torch.save(model.state_dict(), "./weights.pth")

# Load the model
model.load_state_dict(torch.load("./weights.pth"))

# Test the model and save the predcition with ids 
with torch.no_grad():
    test_output = model(test_data)
    _, predicted = torch.max(test_output.data, 1)

    # Create a dataframe with two columns: `id` and `label`
    submission = pd.DataFrame({'id': test['id'], 'label': predicted})
    # submission.to_csv("./data/submission.csv", index=False, columns=['id', 'label'], header=True)