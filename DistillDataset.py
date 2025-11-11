import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from config import Config
import torchvision as tv
from utils import get_transform_CIFAR_10


class Embedder(Dataset):
  def __init__(self, dataset, config):
    super(Embedder, self).__init__()
    self.dataset, self.config = dataset, config
    self.is_consolidated = False
  # __init__

  def __len__(self): return len(self.dataset)

  def __getitem__(self, item):
    if self.is_consolidated: return self.dataset[item][0], self.dataset[item][1]
    feature, label = self.dataset[item]
    patches = feature.unfold(1, 75, 75).unfold(2, 75, 75).permute(1, 2, 0, 3, 4)
    flatten_patches = torch.reshape(input=patches, shape=(9, -1))
    return flatten_patches, label
  # __getitem__

  def consolidate(self):
    buffer = list()
    progression = tqdm(self)
    for feature, label in progression: buffer.append((feature, label))
    self.dataset, self.is_consolidated = buffer, True
    return self
  # consolidate
# Embedder


class DistillDataset(Dataset):
  def __init__(self, dataset, teacher, config: Config):
    super(DistillDataset, self).__init__()
    self.config = config
    self.dataset, self.teacher = dataset, teacher
    self.is_consolidate = False
    self.y = list()
  # __init__

  def __len__(self): return len(self.y)
  def __getitem__(self, item): return self.dataset[item][0], self.y[item]

  def consolidate(self):
    self.teacher.eval()
    progression = tqdm(range(self.config.softset_len))
    for feature, label in progression:
      soft_label = self.teacher(feature)
      self.y.append((soft_label, label))
    # for feature label
    self.is_consolidate = True
    return self
  # consolidate
# DistillDataset

def load_CIFAR_10(transform, path='./data', trainset_len=1000, testset_len=500):
  # trainset, testset are provided as torch.nn.utils.dataset
  trainset = tv.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
  trainset_indices = torch.randperm(trainset.__len__()).tolist()[:trainset_len]
  trainset = Subset(dataset=trainset, indices=trainset_indices)
  testset = tv.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
  testset_indices = torch.randperm(testset.__len__()).tolist()[:testset_len]
  testset = Subset(dataset=testset, indices=testset_indices)
  return trainset, testset
#load_CIFAR_10

if __name__ == "__main__":
  from config import Config

  # load dataset from folder
  cifar_10_transform = get_transform_CIFAR_10(input_size=225)
  trainset, testset = load_CIFAR_10(path='./data', transform=cifar_10_transform)

  trainset = Embedder(dataset=trainset, config=Config()).consolidate()
  trainset = DataLoader(dataset=trainset, batch_size=16)
  for feature, label in trainset:
    print(f'batch_size: {16}\nfeature: {feature.shape}\nlabel:{label.shape}')
    break
  # for

# if __name__ == "__main__"