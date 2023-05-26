import torch.nn as nn

class AlexNet(nn.Module):
    """
    A neural network with 5 convolutional layers and 3 fully connected layers.
    Gradually compresses image information into smaller size.

    Summary:
    •   In terms of training time with gradient descent, saturating nonlinearities tanh(x) are much slower than the non-saturating nonlinearity max(0, x)
    •   Train on multiple GPUs, and communicate only at certain layers
    •	Reduce overfitting by overlapping pooling, data augmentation and dropout
    •	Small amount of weight decay was important

    Ammendments:
    •	Input size changed from (b x 3 x 224 x 224) to (b x 3 x 227 x 227), i.e. 55 x 55 dimensions after first convolutional layer

    References:
    https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    https://github.com/dansuh17/alexnet-pytorch
    """

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.net = nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4), # (b x 3 x 227 x 227) -> (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), # Apply LRN after ReLU
            nn.MaxPool2d(kernel_size=3, stride=2), # (b x 96 x 55 x 55) -> (b x 96 x 27 x 27)

            # Convolutional layer 2
            nn.Conv2d(96, 256, 5, padding=2), # (b x 96 x 27 x 27) -> (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(5, 0.0001, 0.75, 2), # Apply LRN after ReLU
            nn.MaxPool2d(3, 2), # (b x 256 x 27 x 27) -> (b x 256 x 13 x 13)

            # Convolutional layer 3
            nn.Conv2d(256, 384, 3, padding=1), # (b x 256 x 13 x 13) -> (b x 384 x 13 x 13)
            nn.ReLU(),

            # Convolutional layer 4
            nn.Conv2d(384, 384, 3, padding=1), # (b x 384 x 13 x 13) -> (b x 384 x 13 x 13)
            nn.ReLU(),

            # Convolutional layer 5
            nn.Conv2d(384, 256, 3, padding=1), # (b x 384 x 13 x 13) -> (b x 256 x 13 x 13)
            nn.ReLU(),
            nn.MaxPool2d(3, 2) # (b x 256 x 13 x 13) -> (b x 256 x 6 x 6)
        )

        self.classifier = nn.Sequential(
            # Fully-connected layer 1
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
            nn.ReLU(),

            # Fully-connected layer 2
            nn.Dropout(0.5, True),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            # Fully-connected layer 3
            nn.Linear(4096, num_classes)
        )

        self.init_bias()

    def init_bias(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)

        # Initialize bias = 1 for convolutional layers 2, 4, and 5
        nn.init.constant_(self.classifier[4].bias, 1)
        nn.init.constant_(self.classifier[10].bias, 1)
        nn.init.constant_(self.classifier[12].bias, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 256 * 6 * 6)

        return self.classifier(x)
