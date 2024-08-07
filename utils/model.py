import torch
import torch.nn as nn

class Model(nn.Module):
    """    
    Attributes:
        dim_in (int): Dimension of the input.
        nh (int): Number of hidden layers.
        dim_h (int): Dimension of each hidden layer.
        dim_out (int): Dimension of the output.
    """
    def __init__(self, dim_in=3, nh=8, dim_h=20, dim_out=1):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            *[nn.Sequential(nn.ReLU(), nn.Linear(dim_h, dim_h)) for _ in range(nh-1)],
            nn.ReLU(),
            nn.Linear(dim_h, dim_out)
        )

    def forward(self, x):
        return self.layers(x)



class NeuralNetworkImage(nn.Module):
    def __init__(self, ndim_features=6):
        super(NeuralNetworkImage, self).__init__()
        
        self.image_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=ndim_features, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(64, 16, kernel_size=5, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
        )

        self.numeric_features = nn.Sequential(
            nn.Linear(ndim_features, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8, 16 * 16),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        self.combined_features = nn.Sequential(
            nn.Linear(16*16*2, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 16),
            nn.Linear(16, 1),
            nn.Dropout(),

        )
    
    def forward(self, x):

        imgs, tabular = x
        imgs = imgs.view(-1, 1, 64, 64)
        x = self.image_features(imgs)
        x = torch.flatten(x, start_dim=1)  
        y = self.numeric_features(tabular)
        z = torch.cat((x, y), dim=1)
        z = self.combined_features(z)
        
        return z
