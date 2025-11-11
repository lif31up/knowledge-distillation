import copy
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from Embedder import load_MNIST_10, Embedder
from config import Config, TEACH_LOAD_FROM, STNDT_SAVE_TO
from model.ViT import ViT
from utils import get_transform_MNIST_10


def distillate(student, teacher, dataset, config, path, device):
  student.to(device)
  teacher.to(device)
  student.train()
  teacher.eval()

  # optim, criterion, scheduler
  optim = torch.optim.Adam(student.parameters(), lr=config.lr, eps=config.eps)
  criterion = nn.CrossEntropyLoss()
  scheduler = lr_scheduler.StepLR(optim, step_size=5, gamma=0.1)

  progression = tqdm(range(config.iters))
  for _ in progression:
    for feature, label in dataset:
      feature, label = feature.to(device), label.to(device)
      soft_label = F.softmax(teacher(feature) / 4.0, dim=-1)
      output = student(feature)
      distill_loss = criterion(output, soft_label)
      student_loss = criterion(output, label)
      loss = (config.lr * distill_loss) + (1 - config.lr) * student_loss
      optim.zero_grad()
      loss.backward()
      optim.step()
    # for feature, label
    scheduler.step()
    progression.set_postfix(loss=loss.item())

    features = {
      "sate": student.state_dict(),
      "config": config
    } # feature
    torch.save(features, f"{path}")
  # for
# distillate

if __name__ == "__main__":
  config = Config()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # load dataset, transform from folder
  mnist_10_transform = get_transform_MNIST_10(input_size=90)
  trainset, testset = load_MNIST_10(path='./data', transform=mnist_10_transform)
  # embed dataset (3 times 3 patches)
  trainset = Embedder(dataset=trainset, config=config).consolidate()
  config.dummy = trainset.__getitem__(0)[0]
  trainset = DataLoader(dataset=trainset, batch_size=config.batch_size)
  testset = Embedder(dataset=testset, config=config).consolidate()
  testset = DataLoader(dataset=testset, batch_size=config.batch_size)
  student_config = copy.deepcopy(config)
  student_config.n_stacks = 3
  student = ViT(config=student_config)
  teacher_data = torch.load(f=TEACH_LOAD_FROM, map_location=torch.device('cpu'), weights_only=False)
  teacher = ViT(config)
  teacher.load_state_dict(teacher_data['sate'])
  distillate(student=student, teacher=teacher, dataset=trainset, config=student_config, path=STNDT_SAVE_TO, device=device)
# if __name__ == "__main__":