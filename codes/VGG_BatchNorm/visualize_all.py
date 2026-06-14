"""
One-shot visualization runner -- called after STEP1 + STEP2 complete.
Generates all figures needed for the report.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch, os, sys, numpy as np
from torch import nn

sys.path.insert(0, '.')
from models.vgg import (VGG_A, VGG_A_BatchNorm, VGG_A_BN_Dropout,
                        VGG_A_Dropout, get_number_of_parameters)
from data.loaders import get_cifar_loader, get_cifar_loader_augmented
from VGG_Loss_Landscape import set_random_seeds, get_accuracy
from visualize import (plot_filters, plot_confusion_matrix,
                       plot_gradient_predictiveness,
                       plot_training_curves_comparison,
                       plot_band_width_metric)

# Paths
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Load CIFAR-10 test set
val_loader = get_cifar_loader(train=False, batch_size=128)
train_loader = get_cifar_loader(train=True, batch_size=128)

criterion = nn.CrossEntropyLoss()

# ============================================================
# 1. Filter Visualization (best model)
# ============================================================
best_model_path = os.path.join(models_path, '04_BN_Dropout.pth')
if os.path.exists(best_model_path):
    print('Generating filter visualization...')
    model = VGG_A_BN_Dropout()
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    plot_filters(model, save_path=os.path.join(figures_path, 'filters.png'))
    plt.close('all')
else:
    # Fallback to any available model
    import glob
    models = glob.glob(os.path.join(models_path, '*.pth'))
    if models:
        print(f'Best model not found, using {models[0]}')
        model = VGG_A_BN_Dropout()
        model.load_state_dict(torch.load(models[0], map_location=device))
        model.to(device)
        plot_filters(model, save_path=os.path.join(figures_path, 'filters.png'))
        plt.close('all')

# ============================================================
# 2. Confusion Matrix
# ============================================================
model_path = os.path.join(models_path, '04_BN_Dropout.pth')
if not os.path.exists(model_path):
    import glob
    models = glob.glob(os.path.join(models_path, '*.pth'))
    model_path = models[0] if models else None

if model_path:
    print('Generating confusion matrix...')
    model = VGG_A_BN_Dropout()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    cm, acc = plot_confusion_matrix(
        model, val_loader, device,
        save_path=os.path.join(figures_path, 'confusion.png'))
    plt.close('all')
    print(f'  Overall accuracy: {acc:.4f}')

# ============================================================
# 3. Gradient Predictiveness (VGG-A vs VGG-A-BN)
# ============================================================
print('Generating gradient predictiveness comparison...')
set_random_seeds(2020, device)
model_no_bn = VGG_A().to(device)
model_bn = VGG_A_BatchNorm().to(device)

# Quick partial training to get non-random weights for both
print('  Briefly training both models for predictiveness measurement...')
opt_no_bn = torch.optim.Adam(model_no_bn.parameters(), lr=1e-3)
opt_bn = torch.optim.Adam(model_bn.parameters(), lr=1e-3)
for i, (x, y) in enumerate(train_loader):
    if i >= 10:  # just 10 batches to move away from random init
        break
    x, y = x.to(device), y.to(device)
    for model, opt in [(model_no_bn, opt_no_bn), (model_bn, opt_bn)]:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

plot_gradient_predictiveness(
    model_no_bn, model_bn, train_loader, device, criterion,
    save_path=os.path.join(figures_path, 'gradient_predictiveness.png'))
plt.close('all')

# ============================================================
# 4. Band Width Analysis (from saved loss landscape curves)
# ============================================================
min_no_bn_path = os.path.join(figures_path, 'min_curve_no_bn.txt')
if os.path.exists(min_no_bn_path):
    print('Generating band width analysis...')
    min_no = np.loadtxt(min_no_bn_path)
    max_no = np.loadtxt(os.path.join(figures_path, 'max_curve_no_bn.txt'))
    min_bn = np.loadtxt(os.path.join(figures_path, 'min_curve_bn.txt'))
    max_bn = np.loadtxt(os.path.join(figures_path, 'max_curve_bn.txt'))
    band_no, band_bn = plot_band_width_metric(
        min_no, max_no, min_bn, max_bn,
        save_path=os.path.join(figures_path, 'band_width_analysis.png'))
    plt.close('all')
    print(f'  Mean band width -- No BN: {band_no.mean():.4f}, With BN: {band_bn.mean():.4f}')
    print(f'  BN reduces band width by {(1 - band_bn.mean()/band_no.mean())*100:.1f}%')

print(f'\nAll visualizations saved to {figures_path}/')
print('Files:')
for f in sorted(os.listdir(figures_path)):
    print(f'  {f}')
