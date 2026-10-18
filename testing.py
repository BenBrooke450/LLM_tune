

import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(5000, 5000, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("Result device:", y.device)
print("Done!")