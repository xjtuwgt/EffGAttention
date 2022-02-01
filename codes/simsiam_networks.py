import torch.nn as nn
from codes.gdt_subgraph_encoder import GDTSubGraphEncoder, RGDTSubGraphEncoder
from torch import Tensor
from codes.gnn_utils import LinearClassifier


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
    The learning rate of backbone and projector should be different (the latter should be larger)
    Why name this as predictor: predict the output of target networks (w/o gradient) based on the output of
    online networks
    Understanding Self-Supervised Learning Dynamics without Contrastive Pairs, ICML
    """
    def __init__(self, backbone: nn.Module, backbone_out_dim: int):
        super(SimSiam, self).__init__()
        # create the encoder = base_encoder + a two-layer projector
        self.prev_dim = backbone_out_dim
        self.hidden_dim = 4 * backbone_out_dim
        self.graph_encoder = backbone
        # build a 2-layer projection
        self.predictor = Projector(model_dim=self.prev_dim, hidden_dim=self.hidden_dim)  # output layer

    def forward(self, x1, x2, cls_or_anchor='cls'):
        """
        Input:
            x1: first views of input graph
            x2: second views of input graph
            cls_or_anchor: 'cls' or 'anchor'
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        # compute features for one view
        z1 = self.graph_encoder(x1, cls_or_anchor)
        z2 = self.graph_encoder(x2, cls_or_anchor)

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC
        return p1, p2, z1.detach(), z2.detach()  # detach means stop-gradient

    def encode(self, x, cls_or_anchor='cls', project: bool = False):
        z = self.graph_encoder(x, cls_or_anchor)
        if project:
            return self.predictor(z)
        else:
            return z


class SimSiamNodeClassification(nn.Module):
    def __init__(self, config):
        super(SimSiamNodeClassification, self).__init__()
        self.config = config
        if self.config.relation_encoder:
            graph_encoder = RGDTSubGraphEncoder(config=config)
        else:
            graph_encoder = GDTSubGraphEncoder(config=config)
        self.out_dim = self.config.hidden_dim
        self.siamModel = SimSiam(backbone=graph_encoder, backbone_out_dim=self.out_dim)
        self.classifier = LinearClassifier(model_dim=self.out_dim, num_of_classes=self.config.num_classes)

    def init_graph_ember(self, ent_emb: Tensor = None, rel_emb: Tensor = None, rel_freeze=False, ent_freeze=False):
        if self.config.relation_encoder:
            self.siamModel.graph_encoder.init_graph_ember(ent_emb=ent_emb, ent_freeze=ent_freeze,
                                                          rel_emb=rel_emb, rel_freeze=rel_freeze)
        else:
            self.siamModel.graph_encoder.init_graph_ember(ent_emb=ent_emb, ent_freeze=ent_freeze)

    def forward(self, batch):
        h = self.siamModel.encode(x=batch, cls_or_anchor=self.config.cls_or_anchor,
                                  project=self.config.siam_project)
        logits = self.classifier(h)
        return logits
