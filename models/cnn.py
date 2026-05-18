import torch.nn as nn

class PosterCNN(nn.Module):

   def __init__(self, num_classes):
       super(PosterCNN, self).__init__()

       # input: (3, 224, 224)

       self.features = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, padding=1),   # (32, 224, 224)
           nn.BatchNorm2d(32),
           nn.ReLU(),
           nn.MaxPool2d(2, 2),                            # (32, 112, 112)

           nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (64, 112, 112)
           nn.BatchNorm2d(64),
           nn.ReLU(),
           nn.MaxPool2d(2, 2),                            # (64, 56, 56)

           nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (128, 56, 56)
           nn.BatchNorm2d(128),
           nn.ReLU(),
           nn.MaxPool2d(2, 2),                            # (128, 28, 28)

           nn.Conv2d(128, 256, kernel_size=3, padding=1), # (256, 28, 28)
           nn.BatchNorm2d(256),
           nn.ReLU(),
           nn.MaxPool2d(2, 2)                             # (256, 14, 14)
       )


       self.classifier = nn.Sequential(
           nn.Flatten(),
           nn.Linear(256 * 14 * 14, 512),
           nn.BatchNorm2d(512),
           nn.ReLU(),
          
           nn.Linear(512, 256),
           nn.BatchNorm2d(256),
           nn.ReLU(),

           nn.Linear(256, num_classes)
       )


   def forward(self, x):
       x = self.features(x)
       x = self.classifier(x)
       return x