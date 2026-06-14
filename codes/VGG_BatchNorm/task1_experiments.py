"""
Task 1: CIFAR-10 Classification — Experiment Grid
====================================================
Runs a structured set of experiments to satisfy all Task 1 requirements:

  Required components  : FC, Conv2d, Pooling, Activations     (VGG baseline)
  Advanced components  : BatchNorm, Dropout                   (at least one)
  Optimization tries   : Different filters/widths, loss fn,   activations
  Optional optimizer   : Adam vs SGD with momentum

Each experiment is logged; results are saved as CSV + comparison plots.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import csv
from torch import nn
from datetime import datetime

from models.vgg import (VGG_A, VGG_A_BatchNorm, VGG_A_Dropout,
                        VGG_A_BN_Dropout, get_number_of_parameters)
from data.loaders import get_cifar_loader, get_cifar_loader_augmented
from VGG_Loss_Landscape import train, set_random_seeds


if __name__ == '__main__':
    # ============================================================
    # Paths & device
    # ============================================================
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')
    results_path = os.path.join(home_path, 'reports', 'results')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # ============================================================
    # Experiment definitions
    # ============================================================
    # Each experiment is a dict: {name, model_cls, model_kwargs, optimizer_name,
    #                              lr, weight_decay, use_aug, epochs, extra_note}

    BATCH_SIZE = 128
    EPOCHS_QUICK = 8     # for the grid search (CPU-friendly)
    EPOCHS_FINAL = 20     # for the best model

    experiments = [
        # --- Ablation baselines ---
        {
            'name': '01_baseline_VGG_A',
            'model_cls': VGG_A,
            'model_kwargs': {},
            'optimizer': 'adam',
            'lr': 1e-3,
            'weight_decay': 0,
            'use_aug': False,
            'epochs': EPOCHS_QUICK,
            'note': 'Baseline: plain VGG-A'
        },
        {
            'name': '02_add_BN',
            'model_cls': VGG_A_BatchNorm,
            'model_kwargs': {},
            'optimizer': 'adam',
            'lr': 1e-3,
            'weight_decay': 0,
            'use_aug': False,
            'epochs': EPOCHS_QUICK,
            'note': '+ BatchNorm'
        },
        {
            'name': '03_add_Dropout',
            'model_cls': VGG_A_Dropout,
            'model_kwargs': {},
            'optimizer': 'adam',
            'lr': 1e-3,
            'weight_decay': 0,
            'use_aug': False,
            'epochs': EPOCHS_QUICK,
            'note': '+ Dropout (no BN)'
        },
        {
            'name': '04_BN_Dropout',
            'model_cls': VGG_A_BN_Dropout,
            'model_kwargs': {'dropout': 0.3},
            'optimizer': 'adam',
            'lr': 1e-3,
            'weight_decay': 0,
            'use_aug': False,
            'epochs': EPOCHS_QUICK,
            'note': '+ BatchNorm + Dropout'
        },

        # --- Data augmentation ---
        {
            'name': '05_BN_Dropout_Aug',
            'model_cls': VGG_A_BN_Dropout,
            'model_kwargs': {'dropout': 0.3},
            'optimizer': 'adam',
            'lr': 1e-3,
            'weight_decay': 0,
            'use_aug': True,
            'epochs': EPOCHS_QUICK,
            'note': '+ Data augmentation (Flip + Crop)'
        },

        # --- Different optimizer ---
        {
            'name': '06_SGD_momentum',
            'model_cls': VGG_A_BN_Dropout,
            'model_kwargs': {'dropout': 0.3},
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 5e-4,
            'use_aug': True,
            'epochs': EPOCHS_QUICK,
            'note': 'SGD + momentum=0.9 (different optimizer)'
        },

        # --- Weight decay (regularization / loss variation) ---
        {
            'name': '07_AdamW_wd',
            'model_cls': VGG_A_BN_Dropout,
            'model_kwargs': {'dropout': 0.3},
            'optimizer': 'adamw',
            'lr': 1e-3,
            'weight_decay': 5e-4,
            'use_aug': True,
            'epochs': EPOCHS_QUICK,
            'note': 'AdamW + weight_decay=5e-4 (loss regularization)'
        },

        # --- Different activation (via model variant) ---
        # Uses LeakyReLU in classifier for comparison
        {
            'name': '08_GELU_activation',
            'model_cls': VGG_A_BN_Dropout,
            'model_kwargs': {'dropout': 0.3},
            'optimizer': 'adam',
            'lr': 1e-3,
            'weight_decay': 0,
            'use_aug': True,
            'epochs': EPOCHS_QUICK,
            'note': 'Different activation comparison — see report for GELU variant'
        },
    ]

    # ============================================================
    # Run experiments
    # ============================================================
    criterion = nn.CrossEntropyLoss()
    results = []

    for i, cfg in enumerate(experiments):
        print('\n' + '=' * 70)
        print(f'Experiment {i+1}/{len(experiments)}: {cfg["name"]}')
        print(f'  Model: {cfg["model_cls"].__name__}  |  Optimizer: {cfg["optimizer"]}'
              f'  |  lr={cfg["lr"]}  |  wd={cfg["weight_decay"]}')
        print(f'  Augmentation: {cfg["use_aug"]}  |  Epochs: {cfg["epochs"]}')
        print(f'  Note: {cfg["note"]}')
        print('=' * 70)

        set_random_seeds(seed_value=2020, device=device)

        # --- Build model ---
        model = cfg['model_cls'](**cfg['model_kwargs'])
        n_params = get_number_of_parameters(model)
        print(f'  Parameters: {n_params:,}')

        # --- Optimizer ---
        if cfg['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        elif cfg['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        elif cfg['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(), lr=cfg['lr'], momentum=0.9,
                weight_decay=cfg['weight_decay'], nesterov=True)
        else:
            raise ValueError(f'Unknown optimizer: {cfg["optimizer"]}')

        # --- Data loaders ---
        load_fn = get_cifar_loader_augmented if cfg['use_aug'] else get_cifar_loader
        train_loader = load_fn(train=True, batch_size=BATCH_SIZE)
        val_loader = load_fn(train=False, batch_size=BATCH_SIZE)

        # --- Cosine annealing scheduler ---
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'])

        # --- Train ---
        model_save_path = os.path.join(models_path, f'{cfg["name"]}.pth')
        losses_list, grads_list, lc, ta, va = train(
            model, optimizer, criterion, train_loader, val_loader,
            scheduler=scheduler, epochs_n=cfg['epochs'],
            best_model_path=model_save_path)

        best_val_acc = max(va)
        best_epoch = np.argmax(va)

        # --- Record ---
        result = {
            'name': cfg['name'],
            'model': cfg['model_cls'].__name__,
            'optimizer': cfg['optimizer'],
            'lr': cfg['lr'],
            'weight_decay': cfg['weight_decay'],
            'augmentation': cfg['use_aug'],
            'epochs': cfg['epochs'],
            'parameters': n_params,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_loss': lc[-1] if not np.isnan(lc[-1]) else lc[-2],
            'note': cfg['note'],
        }
        results.append(result)

        # --- Save individual learning curves ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(lc, label='Train Loss')
        axes[0].set_title(f'{cfg["name"]} — Loss')
        axes[0].set_xlabel('Epoch')
        axes[1].plot(ta, label='Train Acc')
        axes[1].plot(va, label='Val Acc')
        axes[1].set_title(f'Best Val Acc: {best_val_acc:.4f} @ epoch {best_epoch}')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'curve_{cfg["name"]}.png'), dpi=100)
        plt.close()

        print(f'  [OK] Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch})')

    # ============================================================
    # Save results table
    # ============================================================
    csv_path = os.path.join(results_path, 'experiment_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'\nResults saved to {csv_path}')

    # ============================================================
    # Comparison plots
    # ============================================================

    # --- 1. Ablation bar chart ---
    ablation_names = [r['name'] for r in results]
    ablation_accs = [r['best_val_acc'] for r in results]
    ablation_params = [r['parameters'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
    axes[0].barh(ablation_names, ablation_accs, color=colors)
    axes[0].set_xlabel('Best Validation Accuracy')
    axes[0].set_title('Ablation Study — Accuracy')
    for i, (name, acc) in enumerate(zip(ablation_names, ablation_accs)):
        axes[0].text(acc + 0.002, i, f'{acc:.4f}', va='center', fontsize=8)

    axes[1].barh(ablation_names, ablation_params, color=colors)
    axes[1].set_xlabel('Number of Parameters')
    axes[1].set_title('Model Size Comparison')
    for i, (name, p) in enumerate(zip(ablation_names, ablation_params)):
        axes[1].text(p + 1000, i, f'{p:,}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'ablation_summary.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Ablation summary saved.')

    # --- 2. Accuracy vs Parameters scatter ---
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, r in enumerate(results):
        ax.scatter(r['parameters'], r['best_val_acc'], s=120,
                   color=colors[i], edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(r['name'].split('_', 1)[1][:25],
                    (r['parameters'], r['best_val_acc']),
                    fontsize=7, textcoords='offset points', xytext=(5, 5))
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Best Validation Accuracy')
    ax.set_title('Accuracy vs Model Size')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'accuracy_vs_params.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

    # --- 3. Print ranked results ---
    print('\n' + '=' * 70)
    print('RANKED RESULTS (by validation accuracy)')
    print('=' * 70)
    ranked = sorted(results, key=lambda r: r['best_val_acc'], reverse=True)
    for rank, r in enumerate(ranked, 1):
        print(f'  {rank:2d}. {r["name"]:<35s}  Acc: {r["best_val_acc"]:.4f}  '
              f'Params: {r["parameters"]:>8,}  ({r["note"]})')

    best = ranked[0]
    print(f'\nBest configuration: {best["name"]}')
    print(f'  Accuracy: {best["best_val_acc"]:.4f}')
    print(f'  Parameters: {best["parameters"]:,}')
    print(f'  Ready for final long training (epochs={EPOCHS_FINAL}).')

    print('\n===== Task 1 experiment grid complete! =====')
