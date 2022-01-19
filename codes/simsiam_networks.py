import torch.nn as nn
from codes.gdt_subgraph_encoder import GDTEncoder, RGDTEncoder


class Projector(nn.Module):
    def __init__(self, model_dim: int, hidden_dim: int):
        super(Projector, self).__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(self.model_dim, self.hidden_dim, bias=False)
        self.norm_layer = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.w2 = nn.Linear(self.hidden_dim, self.model_dim, bias=False)
        self.init()

    def init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w1.weight, gain=gain)
        nn.init.xavier_normal_(self.w2.weight, gain=gain)

    def forward(self, x):
        return self.w2(self.relu(self.norm_layer(self.w1(x))))


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, backbone: nn.Module, backbone_out_dim: int):
        super(SimSiam, self).__init__()
        # create the encoder = base_encoder + a two-layer projector
        self.prev_dim = backbone_out_dim
        self.hidden_dim = 4 * backbone_out_dim
        self.graph_encoder = backbone
        # build a 2-layer projection
        self.projector = Projector(model_dim=self.prev_dim, hidden_dim=self.hidden_dim)  # output layer

    def forward(self, x1, x2, cls_or_anchor='cls'):
        """
        Input:
            x1: first views of input
            x2: second views of input
            cls_or_anchor: 'cls' or 'anchor'
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        z1 = self.graph_encoder(x1, cls_or_anchor)
        z2 = self.graph_encoder(x2, cls_or_anchor)

        p1 = self.projector(z1)  # NxC
        p2 = self.projector(z2)  # NxC
        return p1, p2, z1.detach(), z2.detach()

    def encode(self, x, cls_or_anchor='cls', project: bool = False):
        z = self.graph_encoder(x, cls_or_anchor)
        if project:
            return self.projector(z)
        else:
            return z


def SimSiam_Builder(config):
    if config.relation_encoder:
        graph_encoder = RGDTEncoder(config=config)
        out_dim = 4 * config.hidden_dim if config.concat else config.hidden_dim
    else:
        graph_encoder = GDTEncoder(config=config)
        out_dim = config.hidden_dim
    simsiam_model = SimSiam(backbone=graph_encoder, backbone_out_dim=out_dim)
    return simsiam_model
