# Focal Loss for PyTorch

<div align="center">
  <img src="https://raw.githubusercontent.com/itakurah/Focal-loss-PyTorch/main/focal_loss_banner.png" width="80%" height="80%" alt="Sitting Posture"> 
</div>

This repository contains an implementation of **Focal Loss**, a modification of cross-entropy loss designed to address class imbalance by focusing on hard-to-classify examples. This implementation is based on the paper [1]:

**[Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002)** 


By Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár

The `focal_loss` class in this repository supports the following tasks:
- **Binary Classification**
- **Multi-class Classification**
- **Multi-label Classification**

## Features

- **Gamma (γ)**: Focusing parameter to reduce the loss for well-classified examples.
- **Alpha (α)**: Class balancing factor to adjust the loss contribution of each class.
- **Reduction**: Specifies the reduction method (`'mean'`, `'sum'`, or `'none'`).

## Focal Loss Overview

Focal Loss modifies the standard cross-entropy loss by adding a modulating factor `(1 - p_t) ** gamma` to focus learning on hard misclassified examples. It's particularly useful for addressing class imbalance in datasets, especially in object detection tasks, as originally described in the paper by Lin et al.

### Mathematical Formulation

The general form of Focal Loss is:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
-   $p_t$​ is the model's estimated probability for the true class:
    -   For correctly classified examples, $p_t$ is high.
    -   For misclassified examples, $p_t$ is low.
-   $\alpha_t$​ is the class balancing factor for the true class $t$.
-   $\gamma$ is the focusing parameter that adjusts the rate at which easy examples are down-weighted.
### Focal Loss Variants

#### 1. **Binary Classification**

For binary classification, where each instance belongs to one of two classes (0 or 1), the Focal Loss becomes:

$$\text{FL}(p) = -\alpha\, [y \cdot (1 - p)^\gamma \log(p) + (1 - y) \cdot p^\gamma \log(1 - p)]$$

-   $p = \sigma(x)$ is the predicted probability after applying the sigmoid function.
-   $y \in \{0, 1\}$ is the ground truth label.
-   $\alpha$ balances the importance of positive and negative examples.

#### 2. **Multi-class Classification**

For multi-class classification with $C$ classes, the Focal Loss is:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

-   $p_t = \text{Softmax}(x)_t$​ is the predicted probability for the true class $t$.
-   $\alpha_t$​ is the class-specific weighting factor.
-   The loss is summed over all classes and averaged over the batch.

#### 3. **Multi-label Classification**

For multi-label classification, where each instance can belong to multiple classes simultaneously, the Focal Loss is applied independently for each class:

$$\text{FL}(p) = -\alpha\, [y \cdot (1 - p)^\gamma \log(p) + (1 - y) \cdot p^\gamma \log(1 - p)]$$

-   $p = \sigma(x)$ is the sigmoid activation applied per class.
-   $y \in \{0, 1\}^C$ is the ground truth label vector.
-   The loss is computed for each class and then summed or averaged.

## Usage

Install the Torch dependency:
- Windows: `pip install -r requirements.txt`
- Linux: `pip3 install -r requirements.txt`

You can use the `FocalLoss` class for different classification tasks by setting the `task_type` argument.

### Binary Classification

```python
import torch
from focal_loss import FocalLoss

criterion = FocalLoss(gamma=2, alpha=0.25, task_type='binary')
inputs = torch.randn(16)  # Logits from the model (batch_size=16)
targets = torch.randint(0, 2, (16,)).float()  # Ground truth (0 or 1)

loss = criterion(inputs, targets)
print(f'Binary Focal Loss: {loss.item()}')
```

### Multi-class Classification

```python
import torch
from focal_loss import FocalLoss

num_classes = 5
alpha = [1.0] * num_classes  # Example class weights
criterion = FocalLoss(gamma=2, alpha=alpha, task_type='multi-class', num_classes=num_classes)
inputs = torch.randn(16, num_classes)  # Logits from the model
targets = torch.randint(0, num_classes, (16,))  # Ground truth labels

loss = criterion(inputs, targets)
print(f'Multi-class Focal Loss: {loss.item()}')
```

### Multi-label Classification

```python
import torch
from focal_loss import FocalLoss

num_classes = 5
criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-label')
inputs = torch.randn(16, num_classes)  # Logits from the model
targets = torch.randint(0, 2, (16, num_classes)).float()  # Ground truth labels

loss = criterion(inputs, targets)
print(f'Multi-label Focal Loss: {loss.item()}')
```

## References
[1] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017). Focal loss for dense object detection. _2017 IEEE International Conference on Computer Vision (ICCV)_. https://doi.org/10.1109/iccv.2017.324                              
