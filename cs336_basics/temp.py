import torch

arr = [torch.tensor([1., 2, 3]), torch.tensor([4., 5, 6]), torch.tensor([7., 8, 9])]
total_norm = torch.norm(torch.stack([torch.norm(p, 2) for p in arr]), 2)
print(total_norm)
