import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from sklearn.multiclass import OneVsRestClassifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Net(nn.Module):
    
    def __init__(self,input_size,class_number):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, class_number)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def NN(x,y,input_size,class_number):

    model = Net(input_size,class_number)
    model=model.double()
    model.to(device)
    x = torch.tensor(list(x)).to(device)
    y = torch.tensor(y).to(device)
    y = y.double()           
    classifier_name="Fully Connected Neural Network"            
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-1)

    for epoch in range(200):
        optimizer.zero_grad()
        output = model(x)        
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print('Loss: {:.3f}'.format(loss.item()))
            
    output[output >= 0.] = 1
    #if output[output < 0.]=0:
    output[output < 0.] = 0
    output
    parameter={'classifier_name':classifier_name,'criterion':criterion, 'optimizer':str(optimizer)}
    return output,parameter