# Regularization Techniques: Dropout and Weight Decay (L2, AdamW)

## Description

**Regularization Techniques** are essential methods in Deep Learning for **preventing *overfitting*** and improving the model's **generalization** by reducing its effective complexity. **Dropout** randomly deactivates neurons during training to force the network to learn more robust representations and avoid co-adaptation. Its variants include **DropConnect**, which deactivates connections (weights) instead of neurons, and **Spatial Dropout**, which deactivates entire channels in CNNs to regularize convolutional layers. **Weight Decay** methods, such as L2 Regularization ($\lambda \sum w^2$), add a penalty to the loss function to encourage smaller weights and simpler models. **AdamW** is an improved implementation of L2 Weight Decay for adaptive optimizers (such as Adam), decoupling the penalty from the gradient update to ensure more correct and effective regularization, being the standard for training large language models (LLMs) and *Transformers* [4].

## Statistics

**Dropout Rate ($p$):** Generally between 0.1 and 0.5; 0.5 is the common value for hidden layers [1]. **Scaling Factor:** Remaining activations are scaled by $1/(1-p)$ during training. **Hyperparameter $\lambda$ (Weight Decay):** Typical values in the range of $10^{-4}$ to $10^{-2}$. **Performance Improvement:** Dropout can reduce the error rate in vision and speech tasks by 1-3% [1]. AdamW demonstrated better generalization performance on *Transformer* models compared to Adam with traditional L2 Regularization [4].

## Features

**Dropout and Variants:** Random deactivation of neurons (Dropout), random deactivation of connections (DropConnect), deactivation of entire channels in CNNs (Spatial Dropout). **Weight Decay:** L1 penalty (*feature* selection), L2 penalty (reduction of weight magnitude), AdamW (decoupled Weight Decay for adaptive optimizers). **Unique Value Proposition:** Reduction of neuron co-adaptation (Dropout), model simplification and smoothing of the decision surface (Weight Decay), correct and effective implementation of Weight Decay in advanced optimizers (AdamW).

## Use Cases

**Computer Vision (CNNs):** Use of **Spatial Dropout** in architectures such as ResNet and VGG. **Natural Language Processing (NLP):** Use of **Dropout** in RNN and LSTM models and, especially, in **Transformer** models (BERT, GPT) to regularize attention layers. **Recommendation Models:** Application of Dropout and DropConnect to prevent memorization of interactions. **Large-Scale Models (LLMs):** The **AdamW** optimizer is the *de facto* standard for training LLMs and vision models due to its superior generalization [4].

## Integration

**TensorFlow/Keras:** Use of the `tf.keras.layers.Dropout` layer and the `kernel_regularizer=l2(lambda)` argument in `Dense` layers. **PyTorch (Spatial Dropout):** Use of `nn.Dropout2d` to deactivate channels in convolutional layers. **PyTorch (AdamW):** Use of the `torch.optim.AdamW` optimizer with the `weight_decay` parameter to apply weight decay in a decoupled and correct way.
```python
# AdamW Example (PyTorch)
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```
```python
# Dropout and L2 Example (Keras)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
model.add(Dense(128, kernel_regularizer=l2(0.001)))
model.add(Dropout(0.5))
```

## URL

http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf