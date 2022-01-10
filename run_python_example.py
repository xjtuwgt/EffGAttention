from torch.nn import InstanceNorm1d
import torch
from codes.utils import seed_everything
insnorm = InstanceNorm1d(1)
seed_everything(seed=42)
x = torch.randn((1, 2, 3))

print(x)
print(insnorm(x))
