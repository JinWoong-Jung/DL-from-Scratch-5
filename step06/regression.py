import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)
x = torch.rand(100, 1)
y = 2 * x + 5 + torch.rand(100, 1)

W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    y = x @ W + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N

lr = 0.1
iters = 100

for i in range(iters):
    y_hat = predict(x)
    loss = mean_squared_error(y_hat, y)

    loss.backward()

    W.data = W.data - lr * W.grad.data
    b.data = b.data - lr * b.grad.data

    W.grad.zero_()
    b.grad.zero_()

    if i % 10 == 0:
        print(f'iter: {i}, loss: {loss.item():.6f}')
    
print(loss.item())
print('=======================')
print('W =', W.item())
print('b =', b.item())

plt.scatter(x.detach().numpy(), y.detach().numpy(), s=10)
x = torch.tensor([[0.0], [1.0]])
y = W.detach().numpy() * x.detach().numpy() + b.detach().numpy()
plt.plot(x, y, color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.show()