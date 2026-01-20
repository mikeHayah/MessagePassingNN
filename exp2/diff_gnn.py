import torch
from torch import nn
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

class Denoiser(nn.Module):
    def __init__(self, in_x, in_e, hidden=128):
        super().__init__()
        mlp = lambda d_in: nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([
            GINConv(mlp(in_x + in_e)),
            GINConv(mlp(hidden + in_e)),
            GINConv(mlp(hidden + in_e)),
        ])
        self.edge_mlp = nn.Sequential(nn.Linear(in_e + hidden, hidden), nn.ReLU(), nn.Linear(hidden, in_e))
        self.node_head = nn.Linear(hidden, in_x)
        self.edge_head = nn.Linear(in_e, in_e)

    def forward(self, x, edge_index, edge_attr, t_emb):
        h = x + t_emb  # simple conditioning; replace with FiLM/concat
        e = edge_attr
        for conv in self.convs:
            h = conv(h, edge_index)  # edge_attr can be concatenated inside conv MLP
            h = nn.functional.relu(h)
        # edge update (simple)
        row, col = edge_index
        e_h = torch.cat([e, h[row]], dim=-1)
        e = self.edge_mlp(e_h)
        return self.node_head(h), self.edge_head(e)

def cosine_beta_schedule(T, s=0.008):
    import math
    steps = torch.arange(T + 1, dtype=torch.float32)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

# Data
ds = QM9(root="data/QM9")
loader = DataLoader(ds, batch_size=64, shuffle=True)

T = 1000
betas = cosine_beta_schedule(T)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    sqrt_ab = alphas_bar[t].sqrt().view(-1, 1)
    sqrt_one_minus_ab = (1 - alphas_bar[t]).sqrt().view(-1, 1)
    return sqrt_ab * x0 + sqrt_one_minus_ab * noise

model = Denoiser(in_x=ds.num_node_features, in_e=ds.num_edge_features)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for data in loader:
    data = data.to("cuda")
    t = torch.randint(0, T, (data.num_graphs,), device=data.x.device)
    noise_x = torch.randn_like(data.x)
    noisy_x = q_sample(data.x, t, noise_x)
    # simple time embedding: broadcast as learned table
    t_emb = nn.functional.embedding(t, torch.randn(T, data.x.size(-1), device=data.x.device))
    pred_x, _ = model(noisy_x, data.edge_index, data.edge_attr, t_emb[data.batch])
    loss = nn.functional.mse_loss(pred_x, noise_x)
    loss.backward()
    opt.step()
    opt.zero_grad()
    break  # remove in real training