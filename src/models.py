import torch
from torch import nn
import torchvision


# NOTE: The final sigmoid layer is removed from the following models because we use the BCEWithLogitsLoss which already
# includes a sigmoid function. We use BCEWithLogitsLoss instead of BCELoss such that we can use a weighted loss function.


class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)

        
class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
        self.head = nn.Conv2d(dec_chs[-1], 1, 1) # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)

        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block

        return self.head(x)  # reduce to 1 channel


class ResNet18UNet(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()
        
        self.freeze = freeze
        if freeze:
            assert pretrained
        
        if pretrained:
            self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            self.resnet = torchvision.models.resnet18()

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Override the final layers
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.freeze:
            self.resnet.eval()
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x = self.resnet.layer4(x3)
        
        x = self.up(self.relu(self.conv1(x))) + x3
        x = self.up(self.relu(self.conv2(x))) + x2
        x = self.up(self.relu(self.conv3(x))) + x1
        x = self.up(self.relu(self.conv4(x)))
        x = self.up(self.relu(self.conv5(x)))
        x = self.conv_out(x)

        return x


class ResNet50UNet(nn.Module):
    def __init__(self, pretrained=False, freeze=False, extra_conv_layers=False):
        super().__init__()
        
        self.freeze = freeze
        if freeze:
            assert pretrained
        
        if pretrained:
            self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        else:
            self.resnet = torchvision.models.resnet50()

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.extra_conv_layers = extra_conv_layers

        # Override the final layers
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        if self.extra_conv_layers:
            self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.conv7 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.freeze:
            self.resnet.eval()
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x = self.resnet.layer4(x3)
        
        x = self.up(self.relu(self.conv1(x))) + x3
        x = self.up(self.relu(self.conv2(x))) + x2
        x = self.up(self.relu(self.conv3(x))) + x1
        x = self.up(self.relu(self.conv4(x)))
        x = self.up(self.relu(self.conv5(x)))

        if self.extra_conv_layers:
            x = self.relu(self.conv6(x)) # no upsampling anymore, already at original input size!
            x = self.relu(self.conv7(x))

        x = self.conv_out(x)

        return x


class ResNet152UNet(nn.Module):
    def __init__(self, pretrained=False, freeze=False, extra_conv_layers=False):
        super().__init__()
        
        self.freeze = freeze
        if freeze:
            assert pretrained
        
        if pretrained:
            self.resnet = torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.DEFAULT)
        else:
            self.resnet = torchvision.models.resnet152()

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.extra_conv_layers = extra_conv_layers

        # Override the final layers
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        if self.extra_conv_layers:
            self.conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
            self.conv7 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        if self.freeze:
            self.resnet.eval()
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x = self.resnet.layer4(x3)
        
        x = self.up(self.relu(self.conv1(x))) + x3
        x = self.up(self.relu(self.conv2(x))) + x2
        x = self.up(self.relu(self.conv3(x))) + x1
        x = self.up(self.relu(self.conv4(x)))
        x = self.up(self.relu(self.conv5(x)))

        if self.extra_conv_layers:
            x = self.relu(self.conv6(x)) # no upsampling anymore, already at original input size!
            x = self.relu(self.conv7(x))
            
        x = self.conv_out(x)

        return x


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.5)
        self.lin1 = nn.Linear(256, 10)
        self.dropout2 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.bn1(self.max_pool(self.relu(self.conv1(x))))
        x = self.bn2(self.max_pool(self.relu(self.conv2(x))))
        x = self.bn3(self.max_pool(self.relu(self.conv3(x))))
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.relu(self.lin1(x))
        x = self.dropout2(x)
        x = self.lin2(x)
        return x


class DilatedResBlock(nn.Module):
    # A repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    # With a skip connection as in resnet blocks, and the possibility to choose the dilation of conv layers
    def __init__(self, in_ch, out_ch, dil_1=1, dil_2=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=dil_1, dilation=dil_1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=dil_2, dilation=dil_2),
                                   nn.ReLU())
        self.skip = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class DilatedResNet18UNet(nn.Module):
    # Replace each of the two last layers of the encoder with two blocks of DilatedResBlock
    def __init__(self, pretrained=False, freeze=False):
        super().__init__()
        
        self.freeze = freeze
        if freeze:
            assert pretrained
        
        # ResNet-18, optionally pretrained in ImageNet
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Change final 2 groups of convolution
        self.l3_b1 = DilatedResBlock(128, 256, dil_1=1, dil_2=2)
        self.l3_b2 = DilatedResBlock(256, 256, dil_1=2, dil_2=2)

        self.l4_b1 = DilatedResBlock(256, 512, dil_1=2, dil_2=4)
        self.l4_b2 = DilatedResBlock(512, 512, dil_1=4, dil_2=4)
        
        # Decoder
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 12 -> 24
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 24 -> 48
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 48 -> 96
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 96 -> 192
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 192 -> 384
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1) # 384 -> 384
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Sigmoid()
      
    def forward(self, x):
        if self.freeze:
            self.resnet.eval()
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # 384x384x32 -> 192x192x32

        x1 = self.resnet.layer1(x) # 192x192x32 -> 96x96x64
        x2 = self.resnet.layer2(x1) # 96x96x64 -> 48x48x128

        x = self.l3_b1(x2) # 48x48x128 -> 48x48x256
        x3 = self.l3_b2(x) # 48x48x256 -> 48x48x256
        x = self.l4_b1(x3) # 48x48x256 -> 48x48x512
        x = self.l4_b2(x) # 48x48x512 -> 48x48x512
        
        x = self.relu(self.conv1(x)) + x3 # 48x48x512 -> 48x48x256
        x = self.relu(self.conv2(x)) + x2 # 48x48x256 -> 48x48x128
        x = self.up(self.relu(self.conv3(x))) + x1 # 48x48x128 -> 96x96x64
        x = self.up(self.relu(self.conv4(x))) # 96x96x64 -> 192x192x32
        x = self.up(self.relu(self.conv5(x))) # 192x192x32 -> 384x384x16

        x = self.conv_out(x)
        x = self.head(x)
        return x


class DilatedBottleneck(nn.Module):
    # a repeating structure composed of convolutional layers which are all summed up at the output
    def __init__(self, nb_channels, nb_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=3, padding=2**i, dilation=2**i)
            for i in range(nb_blocks)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        res = None
        for block in self.blocks:
            x = block(x)
            x = self.relu(x)
            res = x if res is None else res + x
        return res


class DilatedResNet18UNetv2(nn.Module):
    # Replace the two last layers of the encoder with a DilatedBottleneck
    def __init__(self, pretrained=False, freeze=False, nb_dilated_blocks=3):
        super().__init__()
        
        self.freeze = freeze
        if freeze:
            assert pretrained
        
        # ResNet-18, optionally pretrained in ImageNet
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Override the final layers: we want to extract the full 7x7 final feature map
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()

        # Bottleneck
        self.bottleneck = DilatedBottleneck(128, nb_dilated_blocks)
        
        # Decoder
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 48 -> 96
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 96 -> 192
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 192 -> 384
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1) # 384 -> 384
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Sigmoid()
        
    def forward(self, x):
        if self.freeze:
            self.resnet.eval()
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # 384x384x32 -> 192x192x32

        x1 = self.resnet.layer1(x) # 192x192x32 -> 96x96x64
        x2 = self.resnet.layer2(x1) # 96x96x64 -> 48x48x128

        x = self.bottleneck(x2) + x2 # 48x48x128 -> 48x48x128
        
        x = self.up(self.relu(self.conv1(x))) + x1 # 48x48x128 -> 96x96x64
        x = self.up(self.relu(self.conv2(x))) # 96x96x64 -> 192x192x32
        x = self.up(self.relu(self.conv3(x))) # 192x192x32 -> 224x224x16

        x = self.conv_out(x)
        x = self.head(x)
        return x
