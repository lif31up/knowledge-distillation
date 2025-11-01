from torch.utils.data import Dataset
from tqdm import tqdm


class DistillDataset(Dataset):
  def __init__(self, dataset, config):
    super(DistillDataset, self).__init__()
    self.config = config
    self.dataset = dataset
    self.x, self.y = list(), list()
    self.is_consolidate = False
  # __init__

  def __len__(self): return len(self.x)
  def __getitem__(self, item): return self.x[item], self.y[item]

  def consolidate(self):
    progression = tqdm(self.dataset)
    for feature, label in progression:
      feature = self.config.transform(feature)
      label = (self.config.represent(feature), label)
      self.x.append(feature)
      self.y.append(label)
    self.is_consolidate = True
  # consolidate
# DistillDataset