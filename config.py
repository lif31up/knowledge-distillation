import torch
import torchvision as tv
from torch import nn

# path vars
TEACH_SAVE_TO = "./vit.bin"
TEACH_LOAD_FROM = "./vit.bin"
STNDT_SAVE_TO = "./student.bin"
STNDT_LOAD_FROM = "./student.bin"

class Config:
  def __init__(self, is_teacher=False):
    self.iters = 40
    self.batch_size = 16
    self.dataset_len, self.testset_len = 1000, 500

    self.in_channels = 3
    self.hidden_channels = 32
    self.out_features = 10

    self.act = nn.SiLU()
    self.bias = True
    self.dropout = 0.1
    self.eps = 1e-3
    self.betas = (0.9, 0.98)
    self.epochs = 5
    self.lr = 1e-4
    self.clip_grad = False
    self.mask_prob = 0.3
    self.init_weights = init_weights
  # __init__
# Config

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
# init_weights


if __name__ == "__main__":
  config = Config()
  dataset = torch.utils.data.DataLoader(config.trainset, batch_size=config.batch_size, shuffle=True)
# if __name__ == "__main__":