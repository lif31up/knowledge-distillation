import torchvision as tv


def get_transform_CIFAR_10(input_size=225):
  return tv.transforms.Compose([
    # 1. Augmentation for better generalization
    tv.transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(brightness=0.1, contrast=0.1),  # If RGB
    # 2. Resize and ToTensor
    tv.transforms.Resize((input_size, input_size)),
    tv.transforms.ToTensor(),
    # 3. Normalization using ImageNet statistics for pre-trained models
    tv.transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
    ),
  ])  # TRANSFORM
# get_transform_CIFAR_10