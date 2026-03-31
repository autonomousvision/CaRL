import torch
import torch.nn as nn


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(
      in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    # Replaced BatchNorm with GroupNorm (32 groups is standard)
    self.gn1 = nn.GroupNorm(32, planes)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.gn2 = nn.GroupNorm(32, planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion * planes,
                  kernel_size=1, stride=stride, bias=False),
        nn.GroupNorm(32, self.expansion * planes)
      )

  def forward(self, x):
    out = nn.functional.relu(self.gn1(self.conv1(x)))
    out = self.gn2(self.conv2(out))
    out += self.shortcut(x)
    out = nn.functional.relu(out)
    return out


class ResNetRL(nn.Module):
  def __init__(self, in_channel, block, num_blocks):
    super(ResNetRL, self).__init__()
    self.in_planes = 64

    # Standard Stem
    self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.gn1 = nn.GroupNorm(32, 64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # Residual Layers
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    # Apply Orthogonal Initialization
    self._initialize_weights()

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
      layers.append(block(self.in_planes, planes, s))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        # Orthogonal init with the recommended gain for ReLU
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.GroupNorm):
        # Standard initialization for normalization layers
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    out = self.conv1(x)
    out = self.gn1(out)
    out = self.relu(out)
    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    return out

def ResNet18_PPO(in_channel):
  """Returns a ResNet18 model optimized for RL with GroupNorm."""
  return ResNetRL(in_channel, BasicBlock, [2, 2, 2, 2])

def ResNet34_PPO(in_channel):
  """Returns a ResNet34 model optimized for RL with GroupNorm."""
  return ResNetRL(in_channel, BasicBlock, [3, 4, 6, 3])
