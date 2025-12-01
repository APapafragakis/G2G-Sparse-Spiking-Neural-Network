import os
import sys
import argparse

# Make sure the project root is on sys.path so that
# "models", "data", etc. can be imported without issues.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
from torch import nn

from models.dense_snn import DenseSNN
from models.index_snn import IndexSNN
from models.random_snn import RandomSNN
from models.mixer_snn import MixerSNN, MixerSparseLinear
from data.data_fashionmnist import get_fashion_loaders
from utils.encoding import rate_encode

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*aten::lerp.Scalar_out.*"
)

# Device selection (CUDA / DirectML / CPU)
try:
    import torch_directml
    has_dml = True
except ImportError:
    has_dml = False


def select_device():
    """Select compute device (DirectML, CUDA, or CPU)."""
    if has_dml:
        device = torch_directml.device()
        print(f"Using DirectML device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"Using GPU: {gpu_name} | CUDA version: {cuda_version}")
        return device

    device = torch.device("cpu")
    print("No GPU backend available — using CPU.")
    return device


# Shared hyperparameters
batch_size = 256
T = 50
input_dim = 28 * 28
hidden_dim = 1024
# Dense baseline with roughly the same number of parameters
hidden_dim_dense = 447
num_classes = 10
num_epochs = 20
lr = 1e-3

# Global state for Dynamic Sparse Training (DST)
global_step = 0
# Number of training steps between DST updates
UPDATE_INTERVAL = 1000

# Hebbian buffer:
# for each sparse layer we store the latest batch of
# pre- and post-synaptic activations
hebb_buffer = {
    "fc1": None,  # pre: input spikes, post: layer1 spikes
    "fc2": None,  # pre: layer1 spikes, post: layer2 spikes
    "fc3": None,  # pre: layer2 spikes, post: layer3 spikes
}


def build_model(model_name: str, p_inter: float):
    """Build and return the selected model."""
    if model_name == "dense":
        return DenseSNN(input_dim, hidden_dim_dense, num_classes)

    elif model_name == "index":
        return IndexSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )

    elif model_name == "random":
        return RandomSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )

    elif model_name == "mixer":
        return MixerSNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_groups=8,
            p_intra=1.0,
            p_inter=p_inter,
        )

    raise ValueError(f"Unknown model type: {model_name}")


def update_hebb_buffer(input_spikes: torch.Tensor, activations: dict):
    """
    Store the latest batch of pre/post activations for each MixerSparseLinear layer.

    input_spikes: [T, B, input_dim]
    activations:  dict with keys 'layer1', 'layer2', 'layer3',
                  each tensor has shape [T, B, H]
    """
    global hebb_buffer

    # Everything is kept on CPU to keep GPU memory usage lower
    pre_fc1 = input_spikes.detach().cpu()         # [T, B, 784]
    post_fc1 = activations["layer1"].detach().cpu()

    pre_fc2 = activations["layer1"].detach().cpu()
    post_fc2 = activations["layer2"].detach().cpu()

    pre_fc3 = activations["layer2"].detach().cpu()
    post_fc3 = activations["layer3"].detach().cpu()

    hebb_buffer["fc1"] = {"pre": pre_fc1, "post": post_fc1}
    hebb_buffer["fc2"] = {"pre": pre_fc2, "post": post_fc2}
    hebb_buffer["fc3"] = {"pre": pre_fc3, "post": post_fc3}


def compute_ch_matrix(pre_batch: torch.Tensor, post_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute CH(i,j) = cosine similarity between pre and post neuron activation vectors.

    pre_batch:  [T, B, N_in]
    post_batch: [T, B, N_out]

    Returns:
        ch: [N_out, N_in] cosine similarity matrix
    """
    T_steps, B, N_in = pre_batch.shape
    _, _, N_out = post_batch.shape

    # Flatten time and batch into one axis
    pre_flat = pre_batch.reshape(T_steps * B, N_in)     # [TB, N_in]
    post_flat = post_batch.reshape(T_steps * B, N_out)  # [TB, N_out]

    # Each neuron gets a vector of length TB
    pre_vecs = pre_flat.transpose(0, 1)    # [N_in, TB]
    post_vecs = post_flat.transpose(0, 1)  # [N_out, TB]

    eps = 1e-8
    pre_norm = pre_vecs / (pre_vecs.norm(dim=1, keepdim=True) + eps)
    post_norm = post_vecs / (post_vecs.norm(dim=1, keepdim=True) + eps)

    # Cosine similarity: post_norm @ pre_norm^T
    ch = torch.matmul(post_norm, pre_norm.transpose(0, 1))  # [N_out, N_in]
    return ch


def dst_update_layer_three_prune_hebb_growth(
    layer: MixerSparseLinear,
    layer_name: str,
    prune_frac: float,
):
    """
    DST update for a single MixerSparseLinear layer.

    Pruning (three criteria in sequence):
        1) C_P = SET    (magnitude-based pruning on active edges)
        2) C_P = Random (random pruning on active edges)
        3) C_P = C_H    (Hebbian pruning: lowest CH(i,j) are removed)

    Growth:
        C_G = C_H       (Hebbian growth: highest CH(i,j) among inactive edges)

    Total number of dropped and grown connections is kept roughly the same.
    """
    global hebb_buffer

    # We need a stored batch of pre/post activations for this layer
    buf = hebb_buffer.get(layer_name, None)
    if buf is None or "pre" not in buf or "post" not in buf:
        # No activations yet -> skip this DST step
        return

    pre_batch = buf["pre"]   # [T, B, N_in]
    post_batch = buf["post"] # [T, B, N_out]

    # Shapes must match this layer
    if pre_batch.shape[-1] != layer.in_features or post_batch.shape[-1] != layer.out_features:
        return

    # Compute CH on CPU
    ch_cpu = compute_ch_matrix(pre_batch, post_batch)  # [out_features, in_features]

    weight = layer.weight.data
    mask = layer.mask  # on device

    device = weight.device

    # Work on CPU for mask and CH; weights stay on device
    mask_cpu = mask.detach().cpu()
    w_cpu = weight.detach().cpu()

    active_cpu = mask_cpu.bool()
    num_active = active_cpu.sum().item()
    if num_active == 0:
        return

    # Total target pruning based on current number of active connections
    total_to_prune = int(prune_frac * num_active)
    if total_to_prune < 1:
        return

    # Split pruning budget across the three criteria
    # Here we take an equal share for each stage
    base = total_to_prune // 3
    n_set = base
    n_rand = base
    n_hebb = total_to_prune - n_set - n_rand  # whatever remains

    total_pruned = 0

    # --- 1) C_P = SET (magnitude-based pruning) ---
    active_cpu = mask_cpu.bool()
    if n_set > 0 and active_cpu.sum().item() > 0:
        active_weights = w_cpu[active_cpu].abs()
        # If we have fewer active edges than n_set, just prune all of them
        n_set_eff = min(n_set, active_weights.numel())
        if n_set_eff > 0:
            thresh, _ = torch.kthvalue(active_weights, n_set_eff)
            prune_mask_set = active_cpu & (w_cpu.abs() <= thresh)
            # In case of ties we may prune slightly more; that is fine
            num_pruned_set = prune_mask_set.sum().item()
            mask_cpu[prune_mask_set] = 0
            total_pruned += num_pruned_set

    # --- 2) C_P = Random (random pruning on active edges) ---
    active_cpu = mask_cpu.bool()
    if n_rand > 0 and active_cpu.sum().item() > 0:
        active_idx = active_cpu.nonzero(as_tuple=False)  # [N_active, 2]
        n_rand_eff = min(n_rand, active_idx.size(0))
        if n_rand_eff > 0:
            perm = torch.randperm(active_idx.size(0))[:n_rand_eff]
            rand_idx = active_idx[perm]
            mask_cpu[rand_idx[:, 0], rand_idx[:, 1]] = 0
            total_pruned += n_rand_eff

    # --- 3) C_P = C_H (Hebbian pruning: smallest CH among active edges) ---
    active_cpu = mask_cpu.bool()
    if n_hebb > 0 and active_cpu.sum().item() > 0:
        active_scores = ch_cpu[active_cpu]  # CH for currently active edges
        n_hebb_eff = min(n_hebb, active_scores.numel())
        if n_hebb_eff > 0:
            # Threshold for the lowest CH values
            thresh_hebb, _ = torch.kthvalue(active_scores, n_hebb_eff)
            prune_mask_hebb = active_cpu & (ch_cpu <= thresh_hebb)
            num_pruned_hebb = prune_mask_hebb.sum().item()
            mask_cpu[prune_mask_hebb] = 0
            total_pruned += num_pruned_hebb

    # --- Growth: C_G = C_H (Hebbian growth on inactive edges) ---
    active_cpu = mask_cpu.bool()
    inactive_cpu = ~active_cpu
    num_inactive = inactive_cpu.sum().item()
    if num_inactive == 0 or total_pruned == 0:
        # Sync mask back and clear inactive weights
        mask.copy_(mask_cpu.to(device))
        weight[~mask.bool()] = 0.0
        return

    num_grow = min(total_pruned, num_inactive)
    inactive_idx = inactive_cpu.nonzero(as_tuple=False)  # [N_inactive, 2]

    # Scores for inactive edges: CH(i,j)
    inactive_scores = ch_cpu[inactive_cpu]  # 1D tensor
    n_grow_eff = min(num_grow, inactive_scores.numel())
    if n_grow_eff < 1:
        mask.copy_(mask_cpu.to(device))
        weight[~mask.bool()] = 0.0
        return

    # Take top-k CH for growth (largest CH)
    _, top_idx = torch.topk(inactive_scores, k=n_grow_eff, largest=True)
    grow_idx = inactive_idx[top_idx]  # [n_grow_eff, 2]

    mask_cpu[grow_idx[:, 0], grow_idx[:, 1]] = 1

    # Copy final mask back to the layer
    mask.copy_(mask_cpu.to(device))

    # Clear weights for all inactive connections
    weight[~mask.bool()] = 0.0


def dst_step(model: nn.Module, prune_frac: float = 0.025):
    """
    Apply one DST update step to all MixerSparseLinear layers in the model.

    Current setting:
        C_P = SET  -> Random -> C_H
        C_G = C_H
    """
    for name, module in model.named_modules():
        if isinstance(module, MixerSparseLinear):
            # name should end with 'fc1', 'fc2' or 'fc3' inside MixerSNN
            short_name = name.split(".")[-1]
            if short_name in hebb_buffer:
                dst_update_layer_three_prune_hebb_growth(
                    module,
                    short_name,
                    prune_frac,
                )
    print("[DST] step executed")


def train_one_epoch(model, loader, optimizer, device, epoch_idx: int, use_dst: bool):
    """
    Train the model for one epoch on the given DataLoader.

    In dynamic mode with MixerSNN:
        - request activations from the model
        - update Hebbian buffers
        - apply DST every UPDATE_INTERVAL steps
    """
    global global_step
    model.train()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)

        # Encode images into spike trains
        spikes = rate_encode(images, T).to(device)   # [T, B, 784]

        optimizer.zero_grad()

        if use_dst and isinstance(model, MixerSNN):
            # Ask the model to return per-layer activations for Hebbian updates
            spk_counts, activations = model(spikes, return_activations=True)
            update_hebb_buffer(spikes, activations)
        else:
            spk_counts = model(spikes)  # [B, num_classes]

        loss = nn.CrossEntropyLoss()(spk_counts, labels)
        loss.backward()
        optimizer.step()

        # Dynamic Sparse Training step (MixerSNN only)
        if use_dst and isinstance(model, MixerSNN):
            if global_step > 0 and global_step % UPDATE_INTERVAL == 0:
                dst_step(model, prune_frac=0.025)

        global_step += 1

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate the model and return accuracy."""
    model.eval()
    total = 0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        labels = labels.to(device, non_blocking=True)
        spikes = rate_encode(images, T).to(device)
        spk_counts = model(spikes)

        preds = spk_counts.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return acc


@torch.no_grad()
def compute_firing_rates(model, loader, device):
    """
    Compute mean firing rates per hidden layer and overall
    (only hidden layers are considered).
    """
    model.eval()

    total_samples = 0
    l1_sum = None
    l2_sum = None
    l3_sum = None

    for images, _ in loader:
        B = images.size(0)
        total_samples += B

        spikes = rate_encode(images, T).to(device)
        spk_out_sum, hidden_spikes = model(spikes, return_hidden_spikes=True)

        # hidden_spikes["layerX"]: [B, hidden_dim] (sum of spikes over time)
        batch_l1 = hidden_spikes["layer1"].sum(dim=0)  # [hidden_dim]
        batch_l2 = hidden_spikes["layer2"].sum(dim=0)
        batch_l3 = hidden_spikes["layer3"].sum(dim=0)

        if l1_sum is None:
            l1_sum = batch_l1
            l2_sum = batch_l2
            l3_sum = batch_l3
        else:
            l1_sum += batch_l1
            l2_sum += batch_l2
            l3_sum += batch_l3

    denom = T * total_samples
    l1_rate_per_neuron = l1_sum / denom
    l2_rate_per_neuron = l2_sum / denom
    l3_rate_per_neuron = l3_sum / denom

    hidden_concat = torch.cat(
        [l1_rate_per_neuron, l2_rate_per_neuron, l3_rate_per_neuron]
    )

    rates = {
        "layer1_mean": l1_rate_per_neuron.mean().item(),
        "layer2_mean": l2_rate_per_neuron.mean().item(),
        "layer3_mean": l3_rate_per_neuron.mean().item(),
        "overall_hidden_mean": hidden_concat.mean().item(),
    }

    return rates


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SNN on Fashion-MNIST with different connectivity patterns."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dense",
        choices=["dense", "index", "random", "mixer"],
        help="Model type.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=num_epochs,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--p_inter",
        type=float,
        default=0.15,
        help=(
            "Inter-group connection probability p' for sparse models "
            "(Index, Random, Mixer). Ignored for the dense model."
        ),
    )

    parser.add_argument(
        "--sparsity_mode",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help=(
            "Sparsity mode for sparse models: "
            "'static' = only initial structured sparsity, "
            "'dynamic' = apply Dynamic Sparse Training (DST)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()

    print(f"Selected model: {args.model}")
    print(f"Sparsity mode: {args.sparsity_mode}")

    train_loader, test_loader = get_fashion_loaders(batch_size)
    model = build_model(args.model, p_inter=args.p_inter).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    use_dst = (args.sparsity_mode == "dynamic")

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, epoch, use_dst=use_dst)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | test_acc={test_acc:.4f}")

    rates = compute_firing_rates(model, test_loader, device)
    print("Average firing rates (test set):")
    print(f"  Layer 1 mean rate:        {rates['layer1_mean']:.6f}")
    print(f"  Layer 2 mean rate:        {rates['layer2_mean']:.6f}")
    print(f"  Layer 3 mean rate:        {rates['layer3_mean']:.6f}")
    print(f"  Overall hidden mean rate: {rates['overall_hidden_mean']:.6f}")


if __name__ == "__main__":
    main()
