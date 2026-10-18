import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseModulation(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, dropout=0.1):
        super().__init__()
        self.cos_branch = nn.Linear(dim_in, dim_out)
        self.sin_branch = nn.Linear(dim_in, dim_out)
        self.phase_to_angle = nn.Linear(dim_in, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.shortcut = nn.Linear(dim_in, dim_out)

    def forward(self, residual, phase):
        h_cos = self.cos_branch(residual)
        h_sin = self.sin_branch(residual)
        theta = torch.pi * torch.tanh(self.phase_to_angle(phase))


        proj = (h_sin * h_cos).sum(dim=-1, keepdim=True) / (
                (h_cos * h_cos).sum(dim=-1, keepdim=True) + 1e-6)

        h_sin_ortho = h_sin - proj * h_cos
        phase_impact = torch.cos(theta) * h_cos + torch.sin(theta) * h_sin_ortho

        out = self.out_proj(phase_impact)
        out = self.norm(out + self.shortcut(residual))
        return out


class ConditionalFusion(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, dropout=0.1):
        super().__init__()
        self.content_proj = nn.Linear(dim_in, dim_out)
        self.scale_proj = nn.Linear(dim_in, dim_out)
        self.shift_proj = nn.Linear(dim_in, dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)


    def forward(self, x, query):
        content = self.content_proj(x)
        scale = torch.sigmoid(self.scale_proj(query))
        shift = self.shift_proj(query)
        out = scale * content + shift
        out = self.out_proj(out)
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.individual = configs.individual


        self.Q = nn.Parameter(torch.randn(self.cycle_len, self.enc_in), requires_grad=True)
        self.phase = nn.Parameter(torch.randn(self.cycle_len, self.enc_in), requires_grad=True)
        self.amplitude = nn.Parameter(torch.randn(self.cycle_len, self.enc_in), requires_grad=True)

        self.phase_fusion = PhaseModulation(
            dim_in=self.seq_len,
            dim_out=self.d_model,
            hidden_dim=self.d_model,
            dropout=self.dropout
        )

        self.amplitude_fusion = ConditionalFusion(
            dim_in=self.seq_len,
            dim_out=self.d_model,
            hidden_dim=self.d_model,
            dropout=self.dropout
        )

        self.query_proj = nn.Linear(self.seq_len, self.seq_len)
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )


    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        B, T, C = x.shape
        x_input = x.permute(0, 2, 1)

        # cycle information
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.Q[gather_index].permute(0, 2, 1)
        phase = self.phase[gather_index].permute(0, 2, 1)
        amplitude = self.amplitude[gather_index].permute(0, 2, 1)

        # residual information
        residual = x_input - query_input
        phase_impact = self.phase_fusion(residual, phase)
        amplitude_impact = self.amplitude_fusion(residual, amplitude)

        input_proj = self.input_proj(x_input)
        hidden = self.model(phase_impact + amplitude_impact)
        output = self.output_proj(hidden + input_proj)
        output = output.permute(0, 2, 1)


        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output

