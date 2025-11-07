import torch
import torchvision as tv
from torch import nn


class Config:
  def __init__(self, is_teacher=False):
    self.save_to = "./model.bin"

    self.is_teacher = is_teacher  # to train a teacher True else False
    if not self.is_teacher:
      self.model = torch.load('some_path')
      self.represent = lambda x: softoward(x=x, model=self.model, temperature=self.temperature)
    # if not self.is_teacher

    self.iters = 40

    self.in_channels = 3
    self.hidden_channels = 32
    self.out_features = 10

    self.act = nn.SiLU()
    self.bias = True
    self.n_convs = True
    self.dropout = 0.1
    self.eps = 1e-3
    self.betas = (0.9, 0.98)
    self.epochs = 5
    self.batch_size = 16
    self.lr = 1e-4
    self.clip_grad = False
    self.mask_prob = 0.3
    self.init_weights = init_weights
    self.temperature = 4.0

    self.transform = tv.transforms.Compose([
      tv.transforms.Resize((28, 28)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize(mean=[0.5], std=[0.5]),
    ]) # transform
    self.trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
    self.testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
    self.dummy = self.testset[0][0]
  # __init__
# Config

def softoward(x, model, temperature):
  with torch.no_grad(): pred = model(x)
  return F.softmax(pred / temperature, dim=1)
# softoward

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
# init_weights


if __name__ == "__main__":
  config = Config()
  dataset = torch.utils.data.DataLoader(config.trainset, batch_size=config.batch_size, shuffle=True)
# if __name__ == "__main__":