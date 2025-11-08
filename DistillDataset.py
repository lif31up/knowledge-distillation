from torch.utils.data import Dataset
from tqdm import tqdm
from config import Config


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

if __name__ == "__main__":
  from config import Config
  config = Config(is_teacher=True)
  print(f'{config.dummy[0]}')
# if __name__ == "__main__"