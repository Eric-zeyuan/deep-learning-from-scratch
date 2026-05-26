import matplotlib
matplotlib.use('Agg')
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os
import time

np.random.seed(309)

# ============================================================
# Data Loading
# ============================================================

def load_mnist(data_dir=r'.\dataset\MNIST'):
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

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
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000]
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:]
    train_labs = train_labs[10000:]

    train_imgs = train_imgs / train_imgs.max()
    valid_imgs = valid_imgs / valid_imgs.max()
    test_imgs = test_imgs / test_imgs.max()

    return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)


def train_model(model, optimizer, loss_fn, runner, train_set, dev_set,
                num_epochs=5, log_iters=100, eval_interval=50,
                save_dir='./best_models', early_stop_patience=None,
                exp_name='experiment'):
    print(f"\n{'='*60}")
    print(f"Training: {exp_name}")
    print(f"{'='*60}")
    t0 = time.time()
    runner.train(train_set, dev_set, num_epochs=num_epochs, log_iters=log_iters,
                 eval_interval=eval_interval, save_dir=save_dir,
                 early_stop_patience=early_stop_patience)
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s, best val acc: {runner.best_score:.4f}")
    return runner


def evaluate_test(model, loss_fn, metric, test_set):
    X, y = test_set
    logits = model(X)
    loss = loss_fn(logits, y)
    acc = metric(logits, y)
    preds = np.argmax(logits, axis=-1)
    print(f"Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")
    return acc, loss, preds


def save_learning_curve(runner, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.reshape(-1)
    fig.set_tight_layout(1)
    plot(runner, axes)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved to {save_path}")


# ============================================================
# Part A: MLP Baseline
# ============================================================

def run_mlp_baseline(train_set, dev_set, test_set):
    train_X, train_y = train_set
    valid_X, valid_y = dev_set

    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                              milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                scheduler=scheduler, batch_size=32)

    os.makedirs('./results/mlp_baseline', exist_ok=True)

    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=5, log_iters=200, eval_interval=50,
                         save_dir='./results/mlp_baseline',
                         exp_name='MLP Baseline [784,600,10] SGD+MultiStepLR')

    save_learning_curve(runner, './results/mlp_baseline/learning_curve.png')

    test_acc, test_loss, test_preds = evaluate_test(model, loss_fn, nn.metric.accuracy, test_set)
    model.save_model('./results/mlp_baseline/best_model.pickle')

    return runner, model, test_acc


# ============================================================
# Part C Direction 1: Optimization
# ============================================================

def run_optimization_experiments(train_set, dev_set, test_set):
    train_X, train_y = train_set
    valid_X, valid_y = dev_set

    results = {}

    # Exp 1a: SGD (no scheduler, constant lr)
    os.makedirs('./results/opt_sgd', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=5, log_iters=200, eval_interval=50,
                         save_dir='./results/opt_sgd',
                         exp_name='SGD constant LR')
    results['SGD_constant'] = runner
    save_learning_curve(runner, './results/opt_sgd/learning_curve.png')

    # Exp 1b: Momentum SGD (constant lr)
    os.makedirs('./results/opt_momentum', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=model, mu=0.9)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=5, log_iters=200, eval_interval=50,
                         save_dir='./results/opt_momentum',
                         exp_name='Momentum SGD (mu=0.9)')
    results['Momentum'] = runner
    save_learning_curve(runner, './results/opt_momentum/learning_curve.png')

    # Exp 1c: SGD + StepLR
    os.makedirs('./results/opt_steplr', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=1500, gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                scheduler=scheduler, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=5, log_iters=200, eval_interval=50,
                         save_dir='./results/opt_steplr',
                         exp_name='SGD + StepLR')
    results['StepLR'] = runner
    save_learning_curve(runner, './results/opt_steplr/learning_curve.png')

    # Exp 1d: SGD + ExponentialLR
    os.makedirs('./results/opt_explr', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.998)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                scheduler=scheduler, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=5, log_iters=200, eval_interval=50,
                         save_dir='./results/opt_explr',
                         exp_name='SGD + ExponentialLR')
    results['ExponentialLR'] = runner
    save_learning_curve(runner, './results/opt_explr/learning_curve.png')

    return results


# ============================================================
# Part B: CNN
# ============================================================

def run_cnn_experiment(train_set, dev_set, test_set):
    train_X, train_y = train_set
    valid_X, valid_y = dev_set

    # Reshape for CNN: [N, 784] -> [N, 1, 28, 28]
    train_X_cnn = train_X.reshape(-1, 1, 28, 28)
    valid_X_cnn = valid_X.reshape(-1, 1, 28, 28)

    os.makedirs('./results/cnn', exist_ok=True)

    model = nn.models.Model_CNN()
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                              milestones=[800, 2400, 4000], gamma=0.5)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)

    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn,
                                scheduler=scheduler, batch_size=32)

    print(f"\nCNN parameter count: ~9K vs MLP: ~477K")
    t0 = time.time()
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X_cnn, train_y), (valid_X_cnn, valid_y),
                         num_epochs=5, log_iters=200, eval_interval=50,
                         save_dir='./results/cnn',
                         exp_name='CNN')
    elapsed = time.time() - t0
    print(f"CNN training time: {elapsed:.1f}s")

    save_learning_curve(runner, './results/cnn/learning_curve.png')

    test_X_cnn = test_set[0].reshape(-1, 1, 28, 28)
    test_acc, test_loss, test_preds = evaluate_test(model, loss_fn, nn.metric.accuracy,
                                                     (test_X_cnn, test_set[1]))
    model.save_model('./results/cnn/best_model.pickle')

    return runner, model, test_acc, elapsed


# ============================================================
# Part C Direction 2: Regularization
# ============================================================

def run_regularization_experiments(train_set, dev_set, test_set):
    train_X, train_y = train_set
    valid_X, valid_y = dev_set

    results = {}

    # Exp 3a: Baseline (no regularization, no scheduler)
    os.makedirs('./results/reg_baseline', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU')
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=10, log_iters=300, eval_interval=50,
                         save_dir='./results/reg_baseline',
                         exp_name='Baseline (no reg, 10 epochs)')
    results['Baseline'] = runner
    model.save_model('./results/reg_baseline/best_model.pickle')

    # Exp 3b: Weight Decay (L2)
    os.makedirs('./results/reg_wd', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU', [1e-4, 1e-4])
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=10, log_iters=300, eval_interval=50,
                         save_dir='./results/reg_wd',
                         exp_name='L2 Weight Decay (lambda=1e-4)')
    results['WeightDecay'] = runner
    model.save_model('./results/reg_wd/best_model.pickle')

    # Exp 3c: Dropout (insert Dropout after hidden layer)
    os.makedirs('./results/reg_dropout', exist_ok=True)
    class Model_MLP_Dropout(nn.op.Layer):
        def __init__(self):
            self.layers = [
                nn.op.Linear(784, 600),
                nn.op.ReLU(),
                nn.op.Dropout(p=0.5),
                nn.op.Linear(600, 10)
            ]
        def forward(self, X):
            out = X
            for layer in self.layers:
                out = layer(out)
            return out
        def backward(self, loss_grad):
            g = loss_grad
            for layer in reversed(self.layers):
                g = layer.backward(g)
            return g
        def save_model(self, save_path):
            param_list = []
            for layer in self.layers:
                if layer.optimizable:
                    param_list.append({'W': layer.params['W'], 'b': layer.params['b']})
            with open(save_path, 'wb') as f:
                pickle.dump(param_list, f)
        def load_model(self, param_list):
            with open(param_list, 'rb') as f:
                saved = pickle.load(f)
            idx = 0
            for layer in self.layers:
                if layer.optimizable:
                    layer.params['W'] = saved[idx]['W']
                    layer.params['b'] = saved[idx]['b']
                    layer.W = layer.params['W']
                    layer.b = layer.params['b']
                    idx += 1

    model = Model_MLP_Dropout()
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=10, log_iters=300, eval_interval=50,
                         save_dir='./results/reg_dropout',
                         exp_name='Dropout (p=0.5)')
    results['Dropout'] = runner
    model.save_model('./results/reg_dropout/best_model.pickle')

    # Exp 3d: Early Stopping
    os.makedirs('./results/reg_earlystop', exist_ok=True)
    model = nn.models.Model_MLP([784, 600, 10], 'ReLU')
    optimizer = nn.optimizer.SGD(init_lr=0.06, model=model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=10)
    runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=32)
    runner = train_model(model, optimizer, loss_fn, runner,
                         (train_X, train_y), (valid_X, valid_y),
                         num_epochs=10, log_iters=300, eval_interval=50,
                         save_dir='./results/reg_earlystop',
                         early_stop_patience=10,
                         exp_name='Early Stopping (patience=10)')
    results['EarlyStopping'] = runner
    model.save_model('./results/reg_earlystop/best_model.pickle')

    return results


# ============================================================
# Summary Table
# ============================================================

def print_summary(results_dict):
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Best Val Acc':<15} {'Test Acc':<15}")
    print("-" * 60)
    for name, data in results_dict.items():
        if isinstance(data, dict):
            acc = data.get('val_acc', 'N/A')
            tacc = data.get('test_acc', 'N/A')
        else:
            acc = f"{data:.4f}" if isinstance(data, float) else 'N/A'
            tacc = 'N/A'
        print(f"{name:<30} {str(acc):<15} {str(tacc):<15}")
    print("=" * 70)


# ============================================================
# Main entry point
# ============================================================

if __name__ == '__main__':
    import sys

    print("Loading MNIST data...")
    train_set, dev_set, test_set = load_mnist()

    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    summary = {}

    if mode in ['all', 'mlp']:
        runner_mlp, model_mlp, test_acc_mlp = run_mlp_baseline(train_set, dev_set, test_set)
        summary['MLP_Baseline'] = {'val_acc': runner_mlp.best_score, 'test_acc': test_acc_mlp}

    if mode in ['all', 'optimization', 'opt']:
        opt_results = run_optimization_experiments(train_set, dev_set, test_set)
        for name, runner in opt_results.items():
            summary[f'Opt_{name}'] = {'val_acc': runner.best_score, 'test_acc': 'N/A'}

    if mode in ['all', 'cnn']:
        runner_cnn, model_cnn, test_acc_cnn, cnn_time = run_cnn_experiment(train_set, dev_set, test_set)
        summary['CNN'] = {'val_acc': runner_cnn.best_score, 'test_acc': test_acc_cnn,
                          'train_time': f'{cnn_time:.0f}s'}

    if mode in ['all', 'regularization', 'reg']:
        reg_results = run_regularization_experiments(train_set, dev_set, test_set)
        for name, runner in reg_results.items():
            summary[f'Reg_{name}'] = {'val_acc': runner.best_score, 'test_acc': 'N/A'}

    print_summary(summary)
