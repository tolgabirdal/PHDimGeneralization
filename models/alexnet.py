import torch
import torch.nn as nn

# This is a copy from online repositories 
class AlexNet(nn.Module):

    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64, num_classes=1000):
        # ch is the scale factor for number of channels
        super(AlexNet, self).__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.size = self.get_size()
        a = torch.tensor(self.size).float()
        b = torch.tensor(2).float()
        self.width = int(a) * int(1 + torch.log(a) / torch.log(b))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.size, self.width),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.width, self.width),
            nn.ReLU(inplace=True),
            nn.Linear(self.width, num_classes),
        )

    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        print(y.size())
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet(**kwargs):
    return AlexNet(**kwargs)