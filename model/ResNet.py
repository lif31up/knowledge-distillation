from torch import nn


class ResNet(nn.Module):
  def __init__(self, config):
    super(ResNet, self).__init__()
    self.config = config

    self.act = nn.SiLU()
    self.flatten = nn.Flatten(start_dim=1)
    self.convs = self._get_convs(self.config.n_convs)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc = self._get_fc(self.config.dummy)
  # __init__

  def _get_convs(self, n_convs):
    layers = nn.ModuleList()
    layers.append(PA_ResBlock(in_channels=3, out_channels=self.config.hidden_channels, config=self.config))
    for i in range(1, n_convs):
      layers.append(PA_ResBlock(in_channels=self.hidden_channels, out_channels=self.config.hidden_channels, config=self.config))
    return layers
  # _get_convs

  def _get_fc(self, dummy):
    for conv in self.convs: dummy = conv(dummy)
    dummy = self.pool(dummy)
    dummy = dummy.flatten(0)
    return nn.Linear(in_features=dummy.shape[0], out_features=self.config.out_features, bias=self.config.bias)
  # _get_fc

  def forward(self, x):
    for conv in self.convs: x = conv(x)
    x = self.pool(x)
    x = self.flatten(x)
    return self.fc(x)
  # forward
# ResNet

class PA_ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, config):
    super(PA_ResBlock, self).__init__()
    self.config = config
    self.act = self.config.act
    self.dp = nn.Dropout(self.config.dropout)
    self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=self.config.bias)
  # __init__

  def forward(self, x):
    x = self.act(x)
    x = self.conv(x)
    x = self.dp(x)
    return x
  # forward
# ResBlock


if __name__ == "__main__":
  from config import Config
  resnet = ResNet(config=Config(is_teacher=True))
  print(resnet)
# if __name__ == "__main__":