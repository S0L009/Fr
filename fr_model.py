import torch.nn as nn

class Fr_conv(nn.Module):
    def __init__(self, ip, op) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ip, out_channels=op, kernel_size=3, padding='same'),
            nn.BatchNorm2d(op),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=op, out_channels=op, kernel_size=1, padding='same'),
            nn.BatchNorm2d(op),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.conv(x)
    

class Fr(nn.Module):
    def __init__(self,) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Fr_conv(3, 32),
            Fr_conv(32, 64),
            Fr_conv(64, 64),
        )

        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
            nn.Linear(in_features=25600, out_features=8192,),
            nn.Linear(in_features=8192, out_features=1024),
            nn.Linear(in_features=1024, out_features=512),
        )
    
    def forward(self, x):
        return self.dense(self.flatten(self.conv(x)))