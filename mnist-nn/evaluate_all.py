import matplotlib
matplotlib.use('Agg')
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import os
import pickle

np.random.seed(309)

def load_test_set():
    test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
    test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    return test_imgs / 255.0, test_labs


def evaluate_mlp(model_path, test_X, test_y, is_dropout=False):
    if is_dropout:
        # Dropout model saved with custom format: [layer1_dict, layer2_dict]
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        class Model_MLP_Dropout(nn.op.Layer):
            def __init__(self):
                self.layers = [nn.op.Linear(784, 600), nn.op.ReLU(), nn.op.Dropout(p=0.5), nn.op.Linear(600, 10)]
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
        model = Model_MLP_Dropout()
        for i, layer in enumerate(model.layers):
            if layer.optimizable:
                li = 0 if i == 0 else 1
                layer.params['W'] = saved[li]['W']
                layer.params['b'] = saved[li]['b']
                layer.W = layer.params['W']
                layer.b = layer.params['b']
    else:
        model = nn.models.Model_MLP()
        model.load_model(model_path)
    logits = model(test_X)
    acc = nn.metric.accuracy(logits, test_y)
    preds = np.argmax(logits, axis=-1)
    return acc, preds


def evaluate_cnn(model_path, test_X, test_y):
    model = nn.models.Model_CNN()
    model.load_model(model_path)
    test_X_cnn = test_X.reshape(-1, 1, 28, 28)
    logits = model(test_X_cnn)
    acc = nn.metric.accuracy(logits, test_y)
    preds = np.argmax(logits, axis=-1)
    return acc, preds


if __name__ == '__main__':
    test_X, test_y = load_test_set()
    print(f"Test set: {test_X.shape[0]} samples\n")

    results = {}

    paths = {
        'MLP Baseline (SGD+MultiStepLR)': ('mlp', './best_models/best_model.pickle'),
        'Opt: SGD constant LR': ('mlp', './results/opt_sgd/best_model.pickle'),
        'Opt: Momentum SGD': ('mlp', './results/opt_momentum/best_model.pickle'),
        'Opt: StepLR': ('mlp', './results/opt_steplr/best_model.pickle'),
        'Opt: ExponentialLR': ('mlp', './results/opt_explr/best_model.pickle'),
        'Reg: Baseline (10ep)': ('mlp', './results/reg_baseline/best_model.pickle'),
        'Reg: Weight Decay': ('mlp', './results/reg_wd/best_model.pickle'),
        'Reg: Dropout (p=0.5)': ('dropout', './results/reg_dropout/best_model.pickle'),
        'Reg: Early Stopping': ('mlp', './results/reg_earlystop/best_model.pickle'),
        'CNN': ('cnn', './results/cnn/best_model.pickle'),
    }

    print(f"{'Model':<35} {'Test Acc':<12}")
    print("-" * 47)
    for name, (mtype, path) in paths.items():
        if os.path.exists(path):
            if mtype == 'mlp':
                acc, _ = evaluate_mlp(path, test_X, test_y)
            elif mtype == 'dropout':
                acc, _ = evaluate_mlp(path, test_X, test_y, is_dropout=True)
            else:
                acc, _ = evaluate_cnn(path, test_X, test_y)
            print(f"{name:<35} {acc:.4f}")
            results[name] = acc
        else:
            print(f"{name:<35} {'N/A (no model)'}")

    print("-" * 47)
    if results:
        best_name = max(results, key=results.get)
        print(f"\nBest model: {best_name} ({results[best_name]:.4f})")
