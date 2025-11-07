import torch
from torch import nn
from tqdm import tqdm
from config import Config
from torch.optim import lr_scheduler
from model.ResNet import ResNet


def train(model:nn.Module, path, config: Config, trainset, device, is_student=False):
  model.to(device)
  model.train()

  # optim, criterion, scheduler
  optim = torch.optim.Adam(model.parameters(), lr=config.lr, eps=config.eps)
  criterion = nn.CrossEntropyLoss()
  scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

  progression = tqdm(range(config.iters))
  for _ in progression:
    for feature, label in trainset:
      feature, label = feature.to(device, non_blocking=True), label.to(device, non_blocking=True)
      pred = model(feature)
      loss = criterion(pred, label)
      optim.zero_grad()
      loss.backward()
      optim.step()
      progression.set_postfix(loss=loss.item())
    # for feature label
    scheduler.step()
  # for in progression

  features = {
    "sate": model.state_dict(),
    "config": config
  } # feature
  torch.save(features, f"{config.save_to}.bin")
# train

if __name__ == "__main__":
  from torch.utils.data import DataLoader
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  config = Config(is_teacher=True)
  teacher = ResNet(config)
  trainset = DataLoader(dataset=config.trainset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
  train(model=teacher, path='teacher.bin', config=config, trainset=trainset, device=device, is_student=False)
# if __name__ = "__main__":