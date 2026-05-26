import matplotlib
matplotlib.use('Agg')
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(309)


def load_data():
    train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
    train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

    with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

    with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

    idx = np.random.permutation(np.arange(train_imgs.shape[0]))
    train_imgs = train_imgs[idx] / 255.0
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]
    test_imgs = test_imgs / 255.0

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)


def get_predictions(model, X):
    logits = model(X)
    return np.argmax(logits, axis=-1)


def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.matshow(cm, cmap='Blues')
    plt.colorbar(im)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=9)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_title(title)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved to {save_path}')
    return cm


def plot_misclassified(X, y_true, y_pred, save_path, num_samples=25):
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        print('No misclassified samples!')
        return

    n = min(num_samples, len(mis_idx))
    chosen = np.random.choice(mis_idx, n, replace=False)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.reshape(-1)

    for i, idx in enumerate(chosen):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'{y_true[idx]}->{y_pred[idx]}', fontsize=10)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Samples (True -> Predicted)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Misclassified samples saved to {save_path}')


def plot_mlp_weights(model, save_path, num_neurons=100):
    W = model.layers[0].params['W']  # [784, 600]
    n = min(num_neurons, W.shape[1])

    cols = 10
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    axes = axes.reshape(-1)

    for i in range(n):
        w_img = W[:, i].reshape(28, 28)
        vmax = np.max(np.abs(w_img))
        axes[i].imshow(w_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i].axis('off')

    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle('MLP First Layer Weights (28x28 per neuron)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'MLP weights saved to {save_path}')


def plot_cnn_kernels(model, save_path):
    W = model.layers[0].params['W']  # [8, 1, 3, 3]
    out_channels = W.shape[0]

    cols = 4
    rows = (out_channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.reshape(-1)

    for i in range(out_channels):
        kernel = W[i, 0]  # [3, 3]
        vmax = np.max(np.abs(kernel))
        axes[i].imshow(kernel, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Kernel {i}', fontsize=10)
        axes[i].axis('off')

    for i in range(out_channels, len(axes)):
        axes[i].axis('off')

    plt.suptitle('CNN Conv1 Kernels (8 x 3x3)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'CNN kernels saved to {save_path}')


def plot_learning_curve_comparison(runners, labels, save_path, title='Learning Curve Comparison'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(range(len(runners)))

    for i, (runner, label) in enumerate(zip(runners, labels)):
        train_iters = np.arange(len(runner.train_loss))
        dev_iters = np.arange(len(runner.dev_loss)) * (len(train_iters) / max(len(runner.dev_loss), 1))

        axes[0].plot(train_iters, runner.train_loss, color=colors[i], alpha=0.5, linewidth=0.8)
        axes[0].plot(dev_iters, runner.dev_loss, color=colors[i], label=f'{label} (val)', linewidth=1.5)

        axes[1].plot(train_iters, runner.train_scores, color=colors[i], alpha=0.5, linewidth=0.8)
        axes[1].plot(dev_iters, runner.dev_scores, color=colors[i], label=f'{label} (val)', linewidth=1.5)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend(loc='upper right', fontsize=7)

    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend(loc='lower right', fontsize=7)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Comparison plot saved to {save_path}')


# ============================================================
# Main: Run all visualizations
# ============================================================

def main():
    os.makedirs('./results/viz', exist_ok=True)

    print('Loading data...')
    train_set, dev_set, test_set = load_data()
    test_X, test_y = test_set

    # -- MLP visualization --
    mlp_path = './best_models/best_model.pickle'
    if os.path.exists(mlp_path):
        print('\n=== MLP Visualization ===')
        mlp_model = nn.models.Model_MLP()
        mlp_model.load_model(mlp_path)
        mlp_preds = get_predictions(mlp_model, test_X)

        cm = plot_confusion_matrix(test_y, mlp_preds,
                                   './results/viz/mlp_confusion_matrix.png',
                                   'MLP Confusion Matrix')
        plot_misclassified(test_X, test_y, mlp_preds,
                          './results/viz/mlp_misclassified.png')
        plot_mlp_weights(mlp_model, './results/viz/mlp_weights.png')
    else:
        print(f'MLP model not found at {mlp_path}, skipping MLP viz.')

    # -- CNN visualization --
    cnn_path = './results/cnn/best_model.pickle'
    if os.path.exists(cnn_path):
        print('\n=== CNN Visualization ===')
        cnn_model = nn.models.Model_CNN()
        cnn_model.load_model(cnn_path)
        test_X_cnn = test_X.reshape(-1, 1, 28, 28)
        cnn_preds = get_predictions(cnn_model, test_X_cnn)

        cm = plot_confusion_matrix(test_y, cnn_preds,
                                   './results/viz/cnn_confusion_matrix.png',
                                   'CNN Confusion Matrix')
        plot_misclassified(test_X, test_y, cnn_preds,
                          './results/viz/cnn_misclassified.png')
        plot_cnn_kernels(cnn_model, './results/viz/cnn_kernels.png')
    else:
        print(f'CNN model not found at {cnn_path}, skipping CNN viz.')

    print('\nAll visualizations done!')


if __name__ == '__main__':
    main()
