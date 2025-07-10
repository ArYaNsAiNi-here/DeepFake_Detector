import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Should show 12.1 or 12.2
print(torch.cuda.get_device_name(0))