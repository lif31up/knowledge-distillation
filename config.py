import torch
from torchvision.models import resnet101, ResNet101_Weights
import torchvision as tv

class Config:
  def __init__(self):
    self.save_to = "your_path"

    self.n_heads = 12
    self.n_stacks = 1
    self.n_hidden = 2
    self.dim = 768
    self.output_dim = 2
    self.bias = True

    self.dropout = 0.1
    self.attention_dropout = 0.1
    self.eps = 1e-3
    self.betas = (0.9, 0.98)
    self.epochs = 5
    self.batch_size = 16
    self.lr = 1e-4
    self.clip_grad = False
    self.mask_prob = 0.3
    self.init_weights = init_weights
    self.temperature = 4.0

    self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    self.transform = transform
    self.represent = lambda x: softoward(x=x, model=self.model, temperature=self.temperature)
    self.dummy = None
  # __init__
# Config

transform = tv.transforms.Compose([
  tv.transforms.Resize((28, 28)),
  tv.transforms.Grayscale(num_output_channels=1),
  tv.transforms.ToTensor(),
  tv.transforms.Normalize(mean=[0.5], std=[0.5]),
]) # transform

def softoward(x, model, temperature):
  with torch.no_grad(): pred = model(**x)
  return F.softmax(pred / temperature, dim=1)
# softoward

def init_weights(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None: nn.init.zeros_(m.bias)
# init_weights

if __name__ == "__main__":
  config = Config()
  textset, textset_for_test = get_textset()
  dataset = DistillDataset(textset, config.tokenizer, config.model, config.dim)
  for feature, (soft_target, hard_target) in dataset:
    print(f'feature: {feature}, soft_target: {soft_target}, hard_target: {hard_target}')
    break
  #for feature, (soft_target, hard_target) in dataset:
#if