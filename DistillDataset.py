from torch.utils.data import Dataset
from tqdm import tqdm


class DistillDataset(Dataset):
  def __init__(self, dataset, represent):
    super(DistillDataset, self).__init__()
    self.dataset, self.represent = dataset, represent
    self.x, self.y = list(), list()
    self.is_consolidate = False
  # __init__

  def __len__(self): return len(self.y)
  def __getitem__(self, item): return self.x[item], self.y[item]

  def consolidate(self):
    progression = tqdm(self.dataset)
    for feature, label in progression:
      label = (self.represent(feature), label)
      self.x.append(feature)
      self.y.append(label)
    self.is_consolidate = True
    return self
  # consolidate
# DistillDataset

if __name__ == "__main__":
  from config import Config
  config = Config()
  trainset = DistillDataset(dataset=config.trainset, represent=config.represent).consolidate()
  for feature, label in trainset:
    print(f'{feature, label}')
    break
# if __name__ == "__main__"