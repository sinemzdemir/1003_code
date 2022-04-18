import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self, input_size, class_number):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, class_number)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net(20, 3)
model.to(device)


data_path="/media/DATA/home/sinem/tekli_datalar/biological_process_data_combinations/biological_process_ksep_dataframe_Low_Shallow.pkl"
pkl_file = open(data_path, 'rb')
readed_dataset = pickle.load(pkl_file)
pkl_file.close()

data_len=0
integrated_dataframe=readed_dataset
    
integrated_dataframe.columns=['Entry', 'Label', 'Aspect', 'Vector']
integrated_dataframe=integrated_dataframe.drop(['Aspect'],axis=1)
data_len=len(integrated_dataframe['Vector'][0])


label_list = list(integrated_dataframe['Label'])
protein_representation = integrated_dataframe.drop(["Label", "Entry"], axis=1)
proteins=list(integrated_dataframe['Entry'])
vectors=list(protein_representation['Vector'])
protein_and_representation_dictionary=dict(zip(proteins,vectors ))
mlt = MultiLabelBinarizer()
model_label = mlt.fit_transform(label_list)
    
protein_representation_array = np.array(list(protein_representation['Vector']), dtype=float)
model_label_array = np.array(model_label)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)

for epoch in range(200):
    optimizer.zero_grad()
    output = model(protein_representation['Vector'])
    loss = criterion(output, model_label_array)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
      print('Loss: {:.3f}'.format(loss.item()))

output[output >= 0.] = 1
output[output < 0.] = 0


