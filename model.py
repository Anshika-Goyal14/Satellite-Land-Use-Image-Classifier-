import torch 
import torch.nn as nn 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 =nn.Sequential(
        nn.Conv2d(3,32,3,padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2))

        self.block2 = nn.Sequential(
         nn.Conv2d(32,64,3, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2))

        self.block3 = nn.Sequential(
            nn.Conv2d(64,128,3, padding = 1),
         nn.BatchNorm2d(128),
         nn.ReLU(),
        nn.MaxPool2d(2))

        self.block4 = nn.Sequential(
     nn.Conv2d(128,256,3, padding = 1),
  nn.BatchNorm2d(256),
  nn.ReLU(),
 nn.MaxPool2d(2))
        
        self.fc1 = nn.Linear(16384, 512)
        self.reluc_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512,21)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.reluc_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)



if __name__ == "__main__":
    model = Model()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(out.shape)




