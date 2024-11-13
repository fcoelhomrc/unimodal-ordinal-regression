from src.losses import  RoundedReLU
import torch
import torch.nn.functional as F



N = 8
K_outputs = 2
K_classes = 12

A = 5
B = -A/2
x = A * torch.rand(size=(N, K_classes)) + B

y = torch.randint(0, K_classes, size=(N,))

activation = RoundedReLU.apply(x.clone())
activation.requires_grad_()
nll = F.nll_loss(activation, y)

activation2 = F.relu(x.clone())
activation2.requires_grad_()
nll2 = F.nll_loss(activation2, y)

activation.retain_grad()
nll.retain_grad()

activation2.retain_grad()
nll2.retain_grad()

nll.backward()
nll2.backward()

print(f"x: {x}")
print(f"activation: {activation}")
print(f"activation2: {activation2}")


print(f"activation gradient: {activation.grad}")
print(f"activation2 gradient: {activation2.grad}")
print(f"gradient diff: {torch.abs(activation.grad - activation2.grad).sum()}")

