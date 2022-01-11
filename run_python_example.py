from torch.nn.modules.instancenorm import _InstanceNorm

class InstanceNorm(_InstanceNorm):
    r"""Applies instance normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for
    each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
    """
    def __init__(self, in_channels, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=False):
        super().__init__(in_channels, eps, momentum, affine,
                         track_running_stats)

    def forward(self, x: Tensor) -> Tensor:
        out = F.instance_norm(
            x.t().unsqueeze(0), self.running_mean, self.running_var,
            self.weight, self.bias, self.training
            or not self.track_running_stats, self.momentum, self.eps)
        return out.squeeze(0).t()

from torch.nn import LayerNorm
import torch
from codes.utils import seed_everything
print()
seed_everything(seed=42)
x = torch.randint(0, 100, (5,3)).float()
insnorm = InstanceNorm(3)
layerNorm = LayerNorm(3)

y = insnorm(x)

print(layerNorm(x))

print(x)
print(insnorm(layerNorm(x)))
#
# # x_mean = torch.mean(x, dim=2, keepdim=True)
# # # print(x_mean)
# # x_var = torch.var(x, dim=2, keepdim=True)
# # # print(x_var)
# #
# # x_norm = (x - x_mean)/torch.sqrt(x_var + 1e-6)
# #
# # print(x_norm)
# print(x)
# print(insnorm(x))

# import dgl
# g = dgl.graph(([0, 1], [1, 2]))
# g_2 = dgl.transform.khop_graph(g, 2)
# print(g_2.edges())
# g_2 = dgl.khop_graph(g=g, k=3)
# print(g_2.number_of_edges())
# print(g_2.edges())