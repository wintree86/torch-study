import torch

w = torch.tensor([3], dtype=torch.float32, requires_grad=True)
x = torch.tensor([1], dtype=torch.float32, requires_grad=False)
b = torch.tensor([4], dtype=torch.float32, requires_grad=True) 
t = torch.tensor([4], dtype=torch.float32, requires_grad=False)

lr = 0.001
for i in range(5000):
    
    cost = (t - (w * x + b)) ** 2 
    cost.backward(retain_graph=True)
    
    if i % 100 == 0:
        print(w * 1 + b, w.grad, b.grad) 

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()