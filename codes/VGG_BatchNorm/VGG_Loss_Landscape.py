import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Make sure you are using the right device.
device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    """Compute classification accuracy on a given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, epochs_n=100, best_model_path=None):
    """Train the model, recording per-step loss and gradient norms.

    Returns
    -------
    losses_list : list[list[float]]
        losses_list[epoch][step] — per-step training loss
    grads : list[list[float]]
        grads[epoch][step] — per-step total gradient L2 norm
    learning_curve : list[float]
        per-epoch average training loss
    train_accuracy_curve : list[float]
        per-epoch training accuracy
    val_accuracy_curve : list[float]
        per-epoch validation accuracy
    """
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0.0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []   # [epoch][step]
    grads = []         # [epoch][step]
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        model.train()

        loss_list = []   # per-step loss values this epoch
        grad_list = []   # per-step gradient norms this epoch
        learning_curve[epoch] = 0.0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            # --- record per-step loss (scalar) ---
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()

            loss.backward()

            # --- record total gradient L2 norm ---
            # Using total parameter gradient norm instead of hardcoding
            # classifier[4].weight.grad — the layer index shifts when
            # BatchNorm layers are inserted, making the hardcoded index
            # fragile across VGG_A vs VGG_A_BatchNorm.
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            grad_list.append(total_grad_norm ** 0.5)

            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad_list)

        # --- scheduler step (after optimizer, per PyTorch >= 1.1 convention) ---
        if scheduler is not None:
            scheduler.step()

        # --- epoch-level averages ---
        learning_curve[epoch] /= batches_n

        # --- validation ---
        train_acc = get_accuracy(model, train_loader, device)
        val_acc = get_accuracy(model, val_loader, device)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        # --- save best model ---
        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                torch.save(model.state_dict(), best_model_path)

        # --- live plot ---
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
        axes[0].plot(learning_curve[:epoch + 1])
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[1].plot(train_accuracy_curve[:epoch + 1], label='Train')
        axes[1].plot(val_accuracy_curve[:epoch + 1], label='Val')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        plt.show()
        plt.close('all')  # prevent figure leak that slows training

    print(f'Best val accuracy: {max_val_accuracy:.4f} at epoch {max_val_accuracy_epoch}')
    return losses_list, grads, learning_curve, train_accuracy_curve, val_accuracy_curve


# ============================================================
# Main: Loss Landscape Experiment
# ============================================================
if __name__ == '__main__':
    # Initialize data loaders
    train_loader = get_cifar_loader(train=True)
    val_loader = get_cifar_loader(train=False)
    for X, y in train_loader:
        # Verify one sample
        break

    # Train the same architecture with different learning rates.
    # At each step, the spread between max and min loss across LRs
    # measures landscape smoothness: a narrower band → smoother landscape.
    # BN is expected to produce a visibly narrower band.

    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    epo = 8  # sufficient to show BN smoothing effect; fits deadline
    criterion = nn.CrossEntropyLoss()
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]


    def train_one_config(ModelClass, lr, seed, train_loader, val_loader, epochs, model_save_path=None):
        """Train a single model configuration and return per-step losses."""
        set_random_seeds(seed_value=seed, device=device)
        model = ModelClass()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses_list, grads_list, lc, ta, va = train(
            model, optimizer, criterion, train_loader, val_loader,
            epochs_n=epochs, best_model_path=model_save_path)
        # Flatten per-epoch loss lists into one long per-step list
        flat_losses = []
        for epoch_losses in losses_list:
            flat_losses.extend(epoch_losses)
        return flat_losses, va


    def compute_min_max_curves(all_losses):
        """Given a list of per-step loss lists (one per LR), compute
        element-wise min and max curves.

        All loss lists must have the same length (same # of training steps).
        """
        n_steps = min(len(lst) for lst in all_losses)
        # Truncate all to the same length
        aligned = [lst[:n_steps] for lst in all_losses]
        min_curve = np.min(np.array(aligned), axis=0)
        max_curve = np.max(np.array(aligned), axis=0)
        return min_curve, max_curve


    def plot_loss_landscape(min_curve_no_bn, max_curve_no_bn,
                            min_curve_bn, max_curve_bn,
                            save_path=None):
        """Plot loss landscape comparison: VGG-A vs VGG-A + BN.

        The filled area between min and max curves shows the loss variation
        across different learning rates.  BN makes the landscape smoother,
        so its band should be narrower.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        steps = np.arange(len(min_curve_no_bn))
        # --- No BN ---
        ax = axes[0]
        ax.plot(steps, min_curve_no_bn, color='C0', alpha=0.8, linewidth=0.8)
        ax.plot(steps, max_curve_no_bn, color='C0', alpha=0.8, linewidth=0.8)
        ax.fill_between(steps, min_curve_no_bn, max_curve_no_bn,
                        color='C0', alpha=0.25)
        ax.set_title('VGG-A (without BN)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_ylim(bottom=0)

        # --- With BN ---
        ax = axes[1]
        ax.plot(steps, min_curve_bn, color='C1', alpha=0.8, linewidth=0.8)
        ax.plot(steps, max_curve_bn, color='C1', alpha=0.8, linewidth=0.8)
        ax.fill_between(steps, min_curve_bn, max_curve_bn,
                        color='C1', alpha=0.25)
        ax.set_title('VGG-A with BatchNorm')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')

        fig.suptitle('Loss Landscape Comparison: BN Smooths the Optimization Landscape',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Loss landscape figure saved to {save_path}')
        plt.show()
        plt.close('all')


    # --- Run the experiment ---
    print('=' * 60)
    print('Training VGG-A (without BN) at {} learning rates...'.format(len(learning_rates)))
    print('=' * 60)
    all_losses_no_bn = []
    for i, lr in enumerate(learning_rates):
        print(f'\n--- VGG-A, lr={lr} ({i+1}/{len(learning_rates)}) ---')
        flat, va = train_one_config(VGG_A, lr, seed=2020,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    epochs=epo)
        all_losses_no_bn.append(flat)
        print(f'  Final val accuracy: {va[-1]:.4f}')

    print('\n' + '=' * 60)
    print('Training VGG_A_BatchNorm (with BN) at {} learning rates...'.format(len(learning_rates)))
    print('=' * 60)
    all_losses_bn = []
    for i, lr in enumerate(learning_rates):
        print(f'\n--- VGG_A_BatchNorm, lr={lr} ({i+1}/{len(learning_rates)}) ---')
        flat, va = train_one_config(VGG_A_BatchNorm, lr, seed=2020,
                                    train_loader=train_loader,
                                    val_loader=val_loader,
                                    epochs=epo)
        all_losses_bn.append(flat)
        print(f'  Final val accuracy: {va[-1]:.4f}')

    # --- Compute min/max curves ---
    min_no_bn, max_no_bn = compute_min_max_curves(all_losses_no_bn)
    min_bn, max_bn = compute_min_max_curves(all_losses_bn)

    # --- Plot & save ---
    plot_loss_landscape(min_no_bn, max_no_bn, min_bn, max_bn,
                        save_path=os.path.join(figures_path, 'loss_landscape_comparison.png'))

    # --- Save curves as text for later use ---
    np.savetxt(os.path.join(figures_path, 'min_curve_no_bn.txt'), min_no_bn, fmt='%.6f')
    np.savetxt(os.path.join(figures_path, 'max_curve_no_bn.txt'), max_no_bn, fmt='%.6f')
    np.savetxt(os.path.join(figures_path, 'min_curve_bn.txt'), min_bn, fmt='%.6f')
    np.savetxt(os.path.join(figures_path, 'max_curve_bn.txt'), max_bn, fmt='%.6f')
    print('Curves saved to', figures_path)