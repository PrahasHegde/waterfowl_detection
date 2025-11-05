import torch
print(torch.cuda.is_available())
# Expected output for GPU: True
# Your current output: False

print(torch.version.cuda)
# Expected output for GPU: A version number like '11.8'
# Your current output: None

