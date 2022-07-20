import torch
from torch import nn
from torch import distributions as dist
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
       x = (x - x.mean(dim=self.dim, keepdim=True)) / torch.sqrt(x.var(dim=self.dim, keepdim=True)+self.eps)
       return x


class STPointNetBlock(nn.Module):
    def __init__(self, input_dim, layer_dims, layernorm=True, global_feat=False,
                 transposed=False):
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.layernorm = layernorm
        self.global_feat = global_feat
        self.transposed = transposed
        self.activation = nn.ReLU()

        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, input_dim)

        self.conv_layers = nn.ModuleList()
        for idx in range(len(layer_dims) - 1):
            self.conv_layers.append(nn.Conv1d(layer_dims[idx], layer_dims[idx + 1], 1))
        if layernorm:
            self.ln = LayerNorm(dim=1)

        if not global_feat:
            self.last_conv = nn.Conv1d(layer_dims[-1]*2, layer_dims[-1]*2, 1)

    def forward(self, x):
        if self.transposed:
            batch_size, window_size, num_points = x.size(0), x.size(2), x.size(3)
            x = x.view(batch_size, self.input_dim, window_size*num_points)
        else:
            batch_size, window_size, num_points = x.size(0), x.size(1), x.size(2)
            x = x.permute(0, 3, 1, 2).view(batch_size, self.input_dim, window_size*num_points)

        x = self.activation(self.conv_layers[0](x))
        if self.global_feat is False:
            local_features = x.view(batch_size, -1, window_size, num_points)

        for idx in range(1, len(self.conv_layers) - 1):
            x = self.activation(self.conv_layers[idx](x))

        x = self.conv_layers[-1](x)

        x = x.view(-1, self.layer_dims[-1], window_size, num_points)
        x = torch.max(x, 3)[0]

        if self.global_feat:
            if self.layernorm:
                return self.ln(x)
            return x

        x = x.view(-1, self.layer_dims[-1], window_size, 1).repeat(1, 1, 1, num_points)

        x = torch.cat((x, local_features), dim=1)
        x = x.view(batch_size, -1, window_size*num_points) + self.last_conv(x.view(batch_size, -1, window_size*num_points))
        x = x.view(batch_size, -1, window_size, num_points)
        
        if self.layernorm:
            return self.ln(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)


class TemporalPointAE(nn.Module):
    def __init__(self, input_dim, latent_dim, window_size):
        super().__init__()

        self.latent_dim = latent_dim
        self.window_size = window_size

        self.block_0 = STPointNetBlock(input_dim, [32, 32, 32], transposed=False)
        self.block_1 = STPointNetBlock(64, [64, 64, 64], transposed=True)
        self.block_2 = STPointNetBlock(128, [128, 128, 128], transposed=True)
        self.block_3 = STPointNetBlock(256, [256, 256, 256], transposed=True, global_feat=True)

        self.enc_rnn = nn.GRU(256, 256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_latent = nn.Conv1d(512, latent_dim, 1)
        self.dec_rnn = nn.GRU(latent_dim, 128, batch_first=True, bidirectional=True)

        self.dec_block_0 = STPointNetBlock(6+256, [128, 128, 128], transposed=False)
        self.dec_block_1 = STPointNetBlock(256, [64, 64, 64], transposed=True)
        self.dec_block_2 = STPointNetBlock(128, [32, 32, 32], transposed=True)
        self.dec_block_3 = STPointNetBlock(65, [16, 16, 16], transposed=True)
        
        self.dec_mask = nn.Conv1d(64, 1, 1)
        self.dec_corr = nn.Conv1d(32, 3, 1)
        self.dec_dist = nn.Conv1d(32, 1, 1)

    def encode(self, x):
        batch_size, window_size, num_points = x.size(0), x.size(1), x.size(2)

        x = F.relu(self.block_0(x))
        x = F.relu(self.block_1(x))
        x = F.relu(self.block_2(x))
        x = F.relu(self.block_3(x))

        h_0 = torch.zeros(2, batch_size, 256, dtype=torch.float, device=x.device)

        self.enc_rnn.flatten_parameters()
        x, _ = self.enc_rnn(x.permute(0, 2, 1), h_0)
        x = x.permute(0, 2, 1)

        z = self.fc_latent(x)
        return z

    def decode(self, z, dec_cond):
        batch_size = z.size(0)
        h_0 = torch.zeros(2, batch_size, 128, dtype=torch.float, device=z.device)
        self.dec_rnn.flatten_parameters()
        x, _ = self.dec_rnn(z.permute(0, 2, 1), h_0)

        x = x.unsqueeze(2).repeat(1, 1, dec_cond.size(2), 1)

        x = torch.cat([x, dec_cond], dim=-1)
        window_size, num_points = x.size(1), x.size(2)


        x = F.relu(self.dec_block_0(x))
        x = F.relu(self.dec_block_1(x))
        x = F.relu(self.dec_block_2(x))

        corr_mask = self.dec_mask(x.view(batch_size, -1, window_size*num_points)).view(batch_size, 1, window_size, num_points)
        corr_mask_sigmoid = torch.sigmoid(corr_mask)
        x = torch.cat([x, corr_mask_sigmoid], dim=1)

        x = F.relu(self.dec_block_3(x))
        x = x.view(batch_size, -1, window_size*num_points)
        corr_pts = self.dec_corr(x).view(batch_size, 3, window_size, num_points)
        corr_dist = self.dec_dist(x).view(batch_size, 1, window_size, num_points)

        x = torch.cat([corr_mask, corr_pts, corr_dist], dim=1).permute(0, 2, 3, 1).contiguous()
        return x

    def forward(self, x, dec_cond=None):
        pc_coords = x[..., :3]
        pc_normals = x[..., 8: 11]
        if dec_cond is None:
            dec_cond = torch.cat([pc_coords, pc_normals], dim=-1)

        z = self.encode(x)
        x = self.decode(z, dec_cond)

        recon_obj_mask = x[:, :, :, 0].squeeze(-1)
        recon_obj_corr = x[:, :, :, 1: 4]
        recon_obj_dist = x[:, :, :, 4].squeeze(-1)

        return recon_obj_mask, recon_obj_corr, recon_obj_dist

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
