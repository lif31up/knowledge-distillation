# DistillViT for MNIST-10
This implementation is inspired by:
[Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531) by Geoffrey Hinton, Oriol Vinyals, Jeff Dean.
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2021)](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.

Distillation (or Knowledge Distillation) is a model compression technique where a small model is trained to mimic a large, complex model by learning its "thought process (or soft probabilities)". Most large state-of-the-art models are incredibly accurate but come with high costs—computation, memory, and latency. Distillation captures the knowledge inside those models and packs it into a more efficient one.

- **Task:** Image Recognition
- **Dataset:** MNIST-10

### Requirements
To run the code on your own machine, `run pip install -r requirements.txt`.
```text
tqdm>=4.67.1
```

### Configuration

### Training

### Evaluation