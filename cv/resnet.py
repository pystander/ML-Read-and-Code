import torch.nn as nn

class ResNet(nn.Module):
    """
    A 34-layer deep netowrk that applys residual learning. Inserts shortcut connections to convolutional network.

    Summary:
    •   Deeper plain network has higher training error and test error
    •   Address the degradation problem (not overfitting) by deep residual learning framework
    •   Residual learning: Stop learning x (residual) from shallow layers, but add it to output (shortcut connections)
        (H(x) - x) + x = F(x) + x
    •   Possible explanation: Gradient of F(x) + x is larger (add d[g(x)]/dx to derivation)

    References:
    https://arxiv.org/pdf/1512.03385.pdf
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
    """

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # Convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual network layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Fully-connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        # Add residual learning blocks
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Residual network layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Fully-connected layer
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

class BasicBlock(nn.Module):
    """
    Basic residual learning block with 2 convolutional layers.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False) # Stride = 1
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut connection
        self.downsample = downsample

    def forward(self, x):
        residual = x

        # Convolutional layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Convolutional layer 2
        out = self.conv2(out)
        out = self.bn2(out)

        # Shortcut connection
        if self.downsample is not None:
            residual = self.downsample(x)

        # Residual learning
        out += residual
        out = self.relu(out)

        return out
