import torch
import torchvision as tv
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download


class Config:
  def __init__(self):
    self.save_to = "./model.bin"

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

    self.repo_id = "edadaltocg/resnet50_cifar10"
    self.load_from = "./pytorch_model.bin"
    hf_hub_download(repo_id=self.repo_id, filename=self.load_from)

    self.model = resnet50(weights=None)
    num_ftrs = self.model.fc.in_features
    self.model.fc = torch.nn.Linear(num_ftrs, 10)
    self.model.load_state_dict(torch.load(self.load_from, map_location=lambda storage, loc: storage))

    self.transform = tv.transforms.Compose([
      tv.transforms.Resize((28, 28)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize(mean=[0.5], std=[0.5]),
    ]) # transform
    self.represent = lambda x: softoward(x=x, model=self.model, temperature=self.temperature)

    self.trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
    self.testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
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
  for feature, label in dataset:
    print(f'{config.model(feature)}')
    continue
  # for
# if __name__ == "__main__":