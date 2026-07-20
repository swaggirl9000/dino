import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import SGConv


class ECTBranch(nn.Module):
    def __init__(self, edge_indices: torch.Tensor, out_dim: int = 128):
        super().__init__()
        self.register_buffer("EI", edge_indices)

        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, stride=2, bias=True)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, stride=2, bias=True)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, stride=2, bias=True)
        self.gconv1 = SGConv(128, 128, 39, cached=False, add_self_loops=True)
        self.gconv2 = SGConv(128, 128, 39, cached=False, add_self_loops=True)
        self.linear = nn.Linear(128, out_dim) 

        self.batch_edges = torch.Tensor
        self.n_batch = 0

    def forward(self, x):
        x_shape = x.shape
        if x_shape[0] != self.n_batch:
            self.batch_edges = torch.cat(
                [self.EI + (x_shape[1] * i) for i in range(x_shape[0])], dim=1
            ).to(x.device)
            self.n_batch = x_shape[0]

        x = x.view(x_shape[0] * x_shape[1], 1, -1)

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.adaptive_max_pool1d(x, 1)[..., 0]

        x = self.gconv1(x, self.batch_edges)
        x = F.leaky_relu(x)
        x = self.gconv2(x, self.batch_edges)
        x = F.leaky_relu(x)

        x = x.view(x_shape[0], x_shape[1], -1)
        x = torch.mean(x, dim=1)  # O(3)-invariance

        x = self.linear(x)
        return x  