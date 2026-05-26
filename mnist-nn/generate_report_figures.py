"""Generate missing report figures: CNN learning curve + optimization comparison."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Fig: CNN Learning Curve (from logged eval data)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.set_tight_layout(1)

cnn_eval_iters = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
cnn_eval_accs = [0.813, 0.871, 0.901, 0.9165, 0.9255, 0.9335, 0.9395, 0.945, 0.948]

axes[0].plot(cnn_eval_iters, cnn_eval_accs, 'o-', color='#2E86AB', linewidth=2, markersize=6)
axes[0].set_ylabel('Validation Accuracy')
axes[0].set_xlabel('Iteration')
axes[0].set_title('CNN Validation Accuracy')
axes[0].set_ylim(0.8, 0.97)
axes[0].grid(True, alpha=0.3)

axes[1].text(0.1, 0.7, 'CNN Training Summary', fontsize=14, fontweight='bold')
axes[1].text(0.1, 0.55, 'Architecture: Conv(1,8) -> Pool -> Conv(8,16) -> Pool -> FC(10)', fontsize=10)
axes[1].text(0.1, 0.45, 'Parameters: 9,098 (vs MLP 477,010)', fontsize=10)
axes[1].text(0.1, 0.35, 'Initialization: He (Kaiming) Normal', fontsize=10)
axes[1].text(0.1, 0.25, 'Optimizer: SGD lr=0.06, batch=32', fontsize=10)
axes[1].text(0.1, 0.15, f'Final Val Acc: 94.80%  |  Test Acc: 95.79%', fontsize=10, fontweight='bold')
axes[1].axis('off')

plt.savefig('./results/cnn/learning_curve.png', dpi=120, bbox_inches='tight')
plt.close()
print('[OK] CNN learning curve saved')


# ============================================================
# Fig: Optimization Comparison (4 methods overlaid)
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
labels = ['SGD constant LR', 'Momentum (mu=0.9)', 'StepLR', 'ExponentialLR']
val_accs = [0.9519, 0.9575, 0.9420, 0.9085]
test_accs = [0.9527, 0.9540, 0.9405, 0.9115]
train_times = [50.4, 64.7, 50.6, 50.6]

# Bar chart: validation + test accuracy
x = np.arange(len(labels))
width = 0.35
bars1 = axes[0].bar(x - width/2, val_accs, width, label='Validation', color='#2E86AB', alpha=0.85)
bars2 = axes[0].bar(x + width/2, test_accs, width, label='Test', color='#A23B72', alpha=0.85)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Optimization Methods: Accuracy Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=8)
axes[0].legend()
axes[0].set_ylim(0.88, 0.97)
for bar, val in zip(bars1, val_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}', ha='center', fontsize=7)
for bar, val in zip(bars2, test_accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}', ha='center', fontsize=7)

# Bar chart: training time
bars3 = axes[1].bar(x, train_times, color='#F18F01', alpha=0.85)
axes[1].set_ylabel('Training Time (seconds)')
axes[1].set_title('Optimization Methods: Training Time')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, fontsize=8)
for bar, val in zip(bars3, train_times):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.0f}s', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig('./results/optimization_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print('[OK] Optimization comparison saved')

print('All report figures ready.')
