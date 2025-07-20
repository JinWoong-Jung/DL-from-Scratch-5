import torch

x = torch.tensor(5.0)

y = 3 * x ** 2
print(y)

x = torch.tensor(5.0, requires_grad=True)
y = 3 * x ** 2

y.backward()
print(x.grad)