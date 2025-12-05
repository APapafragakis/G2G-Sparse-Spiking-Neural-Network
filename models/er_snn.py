import torch
from torch import nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


class ERSparseLinear(nn.Module):
    """
    Unstructured Erdos–Rényi sparse linear layer.
    Each weight is independently active with probability p_active.
    """

    def __init__(self, in_features: int, out_features: int, p_active: float, bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_active = float(p_active)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Binary mask sampled once at initialization
        mask = torch.bernoulli(torch.full((out_features, in_features), self.p_active))
        self.register_buffer("mask", mask)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Standard Kaiming init as if the layer was dense
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Rescale active weights per row based on actual fan-in
        with torch.no_grad():
            full_fan_in = float(self.in_features)
            for out_idx in range(self.out_features):
                row_mask = self.mask[out_idx]
                fan_in_active = row_mask.sum().item()
                if 0 < fan_in_active < full_fan_in:
                    scale = (full_fan_in / fan_in_active) ** 0.5
                    self.weight[out_idx, row_mask.bool()] *= scale

        # Bias init based on effective fan-in
        if self.bias is not None:
            fan_in_eff = self.mask.sum(dim=1).float().mean().item()
            if fan_in_eff <= 0:
                fan_in_eff = self.in_features
            bound = 1.0 / fan_in_eff**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply sparse mask on the fly
        effective_weight = self.weight * self.mask
        return F.linear(x, effective_weight, self.bias)


class ERSNN(nn.Module):
    """
    Three-layer SNN with ERSparseLinear layers and LIF neurons.
    Same structure as the other SNN MLPs in the project.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, p_active: float):
        super().__init__()

        self.fc1 = ERSparseLinear(input_dim, hidden_dim, p_active=p_active)
        self.fc2 = ERSparseLinear(hidden_dim, hidden_dim, p_active=p_active)
        self.fc3 = ERSparseLinear(hidden_dim, hidden_dim, p_active=p_active)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        beta = 0.95
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x_seq: torch.Tensor, return_hidden_spikes: bool = False):
        # x_seq: [T, B, input_dim]
        T_steps, B, _ = x_seq.shape

        mem1 = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
        mem2 = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
        mem3 = torch.zeros(B, self.fc3.out_features, device=x_seq.device)
        mem_out = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        spk_out_sum = torch.zeros(B, self.fc_out.out_features, device=x_seq.device)

        if return_hidden_spikes:
            spk1_sum = torch.zeros(B, self.fc1.out_features, device=x_seq.device)
            spk2_sum = torch.zeros(B, self.fc2.out_features, device=x_seq.device)
            spk3_sum = torch.zeros(B, self.fc3.out_features, device=x_seq.device)

        for t in range(T_steps):
            x_t = x_seq[t]

            cur1 = self.fc1(x_t)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_sum += spk_out

            if return_hidden_spikes:
                spk1_sum += spk1
                spk2_sum += spk2
                spk3_sum += spk3

        if return_hidden_spikes:
            hidden_spikes = {
                "layer1": spk1_sum,
                "layer2": spk2_sum,
                "layer3": spk3_sum,
            }
            return spk_out_sum, hidden_spikes

        return spk_out_sum
