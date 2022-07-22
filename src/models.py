import torch
from torch import nn
import torchvision


# NOTE: The final sigmoid layer is removed from the following models because we use the BCEWithLogitsLoss which already
# includes a sigmoid layer. We use BCEWithLogitsLoss instead of BCELoss such that we can use a weighted loss function.


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


class ResnetUNet(nn.Module):
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
      
      self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # 7 -> 14
      self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1) # 14 -> 28
      self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1) # 28 -> 56
      self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1) # 56 -> 112
      self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1) # 112 -> 224
      self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1) # 112 -> 224
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