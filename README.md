# Knowledge Distillation with ResNet
This implementation is inspired by:
[Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531) by Geoffrey Hinton, Oriol Vinyals, Jeff Dean.

Distillation (or Knowledge Distillation) is a model compression technique where a small model is trained to mimic a large, complex model by learning its "thought process (or soft probabilities)". Most large state-of-the-art models are incredibly accurate but come with high costs—computation, memory, and latency. Distillation captures the knowledge inside those models and packs it into a more efficient one.

- **Task:** Image Recognition
- **Dataset:** CIFAR-10


### Requirements
To run the code on your own machine, `run pip install -r requirements.txt`.
```text
tqdm>=4.67.1
```

### Configuration

### Training

### Evaluation