"""
Visualization utilities for Project 2.

- Filter visualization (first conv layer)
- Confusion matrix
- Multi-experiment training curve comparison
- Gradient predictiveness analysis (Task 2 — How BN helps optimization)
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch import nn


# ============================================================
# 1. Convolution Filter Visualization
# ============================================================
def plot_filters(model, save_path=None, title='First-layer Conv Filters'):
    """Visualize the filters of the first Conv2d layer as an RGB grid.

    Each filter (out_channel) is normalized to [0, 1] and displayed
    as a small RGB image patch.
    """
    # Find the first Conv2d layer
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    if first_conv is None:
        print('No Conv2d layer found in model.')
        return

    weights = first_conv.weight.data.cpu().numpy()  # shape: [out_ch, in_ch, h, w]
    out_ch, in_ch, h, w = weights.shape

    # Normalize each filter to [0, 1] for display
    w_min = weights.min(axis=(1, 2, 3), keepdims=True)
    w_max = weights.max(axis=(1, 2, 3), keepdims=True)
    w_range = w_max - w_min
    w_range[w_range == 0] = 1  # avoid div by zero
    weights_norm = (weights - w_min) / w_range

    # Arrange filters in a grid
    n_cols = 8
    n_rows = int(np.ceil(out_ch / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols][i % n_cols] if n_rows > 1 else axes[i % n_cols]
        ax.axis('off')
        if i < out_ch:
            # weights_norm[i] shape: [in_ch, h, w] → transpose to [h, w, in_ch]
            filt = np.transpose(weights_norm[i], (1, 2, 0))
            if in_ch == 1:
                ax.imshow(filt[:, :, 0], cmap='gray')
            else:
                ax.imshow(filt)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Filter visualization saved to {save_path}')
    plt.close()


# ============================================================
# 2. Confusion Matrix
# ============================================================
def plot_confusion_matrix(model, data_loader, device,
                          class_names=None, save_path=None):
    """Compute and plot the confusion matrix on the given data loader."""
    if class_names is None:
        class_names = ['airplane', 'auto', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy()
            targets = y.numpy()
            for t, p in zip(targets, preds):
                cm[t, p] += 1

    # Normalize by row (true class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm[np.isnan(cm_norm)] = 0

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    # Annotate each cell
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.2f}\n({cm[i,j]})',
                    ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Overall accuracy
    acc = np.trace(cm) / np.sum(cm)
    ax.set_title(f'Confusion Matrix (Accuracy: {acc:.4f})')

    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Confusion matrix saved to {save_path}')
    plt.close()

    return cm, acc


# ============================================================
# 3. Training Curve Comparison
# ============================================================
def plot_training_curves_comparison(curves_dict, save_path=None):
    """Plot training/validation curves from multiple experiments on one figure.

    Parameters
    ----------
    curves_dict : dict
        {label: {'train_loss': [...], 'val_acc': [...]}, ...}
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(curves_dict)))
    for (label, curves), color in zip(curves_dict.items(), colors):
        axes[0].plot(curves['train_loss'], label=label, color=color,
                     linewidth=1.2, alpha=0.85)
        axes[1].plot(curves['val_acc'], label=label, color=color,
                     linewidth=1.2, alpha=0.85)

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Training Curve Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Training curves comparison saved to {save_path}')
    plt.close()


# ============================================================
# 4. Gradient Predictiveness (Task 2 — Section "How does BN help?")
# ============================================================
def measure_gradient_predictiveness(model, data_loader, device,
                                     criterion, n_steps=20):
    """Measure how well first-order Taylor expansion predicts loss change.

    At the current model parameters θ:
      1. Compute gradient g on a batch
      2. For step sizes η in logspace(-3, 0, n_steps):
         - Set θ' = θ - η·g
         - Compute L(θ')  (actual loss)
         - Compute L(θ) - η·||g||²  (predicted loss)
      3. Return η values, actual losses, predicted losses

    A smaller gap between actual and predicted → more predictive gradient
    → smoother landscape. BN should reduce this gap.

    Returns
    -------
    etas : np.ndarray   shape (n_steps,)
    actual_losses : np.ndarray  shape (n_steps,)
    predicted_losses : np.ndarray  shape (n_steps,)
    """
    model.eval()
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Get one batch
    x, y = next(iter(data_loader))
    x, y = x.to(device), y.to(device)

    # Compute gradient at current point
    model.zero_grad()
    pred = model(x)
    loss_0 = criterion(pred, y)
    loss_0.backward()

    # Store gradients keyed by name (same keys as state_dict)
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.data.clone()

    # Compute ||g||^2
    g_norm_sq = sum(g.data.norm(2).item() ** 2 for g in grads.values())

    # Test different step sizes
    etas = np.logspace(-3, 0, n_steps)
    actual_losses = np.zeros(n_steps)
    predicted_losses = np.zeros(n_steps)

    for i, eta in enumerate(etas):
        # theta' = theta - eta * g
        for name, p in model.named_parameters():
            if name in grads:
                p.data.copy_(original_state[name] - eta * grads[name])

        # Compute actual loss at θ'
        with torch.no_grad():
            pred_new = model(x)
            actual_losses[i] = criterion(pred_new, y).item()

        # Predicted: L(θ) - η·||g||²
        predicted_losses[i] = loss_0.item() - eta * g_norm_sq

    # Restore original parameters
    model.load_state_dict(original_state)

    return etas, actual_losses, predicted_losses


def plot_gradient_predictiveness(model_no_bn, model_bn, data_loader, device,
                                  criterion, save_path=None):
    """Compare gradient predictiveness for models with and without BN.

    The closer the actual loss curve is to the predicted (straight) line,
    the more predictive the gradient — indicating a smoother landscape.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, model, label in zip(axes,
                                [model_no_bn, model_bn],
                                ['VGG-A (without BN)', 'VGG-A with BatchNorm']):
        etas, actual, predicted = measure_gradient_predictiveness(
            model, data_loader, device, criterion)

        ax.plot(etas, predicted, 'k--', linewidth=1.5, label='Predicted (1st-order Taylor)')
        ax.plot(etas, actual, 'C3-', linewidth=1.5, label='Actual loss')
        ax.fill_between(etas, actual, predicted, alpha=0.2, color='C3')
        ax.set_xscale('log')
        ax.set_xlabel('Step size η')
        ax.set_ylabel('Loss')
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Compute predictiveness error (area between curves)
        error = np.trapezoid(np.abs(actual - predicted), etas)
        ax.text(0.05, 0.95, f'Predictiveness error: {error:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Gradient Predictiveness: BN Makes the Landscape More Linear',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Gradient predictiveness figure saved to {save_path}')
    plt.close()


# ============================================================
# 5. Loss Landscape — Band Width Comparison (quantitative)
# ============================================================
def plot_band_width_metric(min_curve_no_bn, max_curve_no_bn,
                           min_curve_bn, max_curve_bn,
                           save_path=None):
    """Quantitative comparison: the band width (max - min) over training steps.

    A narrower band width for the BN model confirms smoother landscape.
    """
    band_no_bn = max_curve_no_bn - min_curve_no_bn
    band_bn = max_curve_bn - min_curve_bn

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Band width over time
    steps = np.arange(len(band_no_bn))
    axes[0].plot(steps, band_no_bn, 'C0-', alpha=0.7, linewidth=0.8, label='Without BN')
    axes[0].plot(steps, band_bn, 'C1-', alpha=0.7, linewidth=0.8, label='With BN')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Band Width (max - min loss)')
    axes[0].set_title('Loss Landscape Band Width Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram of band widths
    axes[1].hist(band_no_bn, bins=40, alpha=0.5, color='C0', label=f'No BN  (μ={band_no_bn.mean():.3f})')
    axes[1].hist(band_bn, bins=40, alpha=0.5, color='C1', label=f'With BN (μ={band_bn.mean():.3f})')
    axes[1].set_xlabel('Band Width')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Band Width')
    axes[1].legend()

    plt.suptitle('Quantitative Landscape Smoothness Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Band width analysis saved to {save_path}')
    plt.close()

    return band_no_bn, band_bn
