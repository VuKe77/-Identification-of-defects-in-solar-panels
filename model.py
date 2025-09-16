
#%%
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super().__init__()

        #1
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU()

        #2
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU()
        #res-connection
        self.conv1x1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x1 = self.relu1(self.bn1(self.conv1(x))) 
        x1 = self.relu2(self.bn2(self.conv2(x1)))
        #Residual connection
        x = self.bn1x1(self.conv1x1(x))
        x1 = x1 + x
        return x1
    
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,7,2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3,2)
        self.res1 = ResBlock(64,128,2)
        self.res2 = ResBlock(128,256,2)
        self.res3 = ResBlock(256,512,2)
        self.pool2 = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc  = nn.Linear(512,2)
        self.activation = nn.Sigmoid()


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.activation(x)
        #Maybe we need to perform threshold immidiatly here?
        return x

    
#%%
if __name__=="__main__":
    from torchinfo import summary
    input = torch.rand((5,3,20,20))
    m1 = ResBlock(3,20) 
    m1.forward(input)
    m2 = ResNet()
    a2 = m2.forward(input)
    print(a2.shape)
    print(summary(m2,input_data = input))




