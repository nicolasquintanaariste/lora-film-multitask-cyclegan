import torch
# pytorc tensor imaeg tensor is shape (batch_size, channels, height, width)

a =1 
b =2
tensor1 = torch.randn(2, 3, 4, 5)
tensor2 = torch.randn(2, 3, 4, 5)
tensor_sum = tensor1 + tensor2
c = a + b
print(f"Sum: {c}")