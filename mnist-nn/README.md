# MNIST Classification from Scratch (NumPy only)

A pure NumPy implementation of neural networks for MNIST handwritten digit classification. Built as part of a deep learning course project.

## What's Inside

| Component | File | Description |
|-----------|------|-------------|
| Linear Layer | `mynn/op.py` | Forward/backward with L2 weight decay |
| Conv2D (im2col) | `mynn/op.py` | Vectorized 2D convolution, 105x faster than naive loops |
| MaxPool2D, Flatten, Dropout | `mynn/op.py` | CNN building blocks |
| Cross-Entropy Loss | `mynn/op.py` | Softmax + cross-entropy with combined gradient |
| MLP Model | `mynn/models.py` | Configurable depth/width multi-layer perceptron |
| CNN Model | `mynn/models.py` | 2-layer ConvNet with He initialization |
| SGD, Momentum SGD | `mynn/optimizer.py` | Optimizers with weight decay support |
| LR Schedulers | `mynn/lr_scheduler.py` | StepLR, MultiStepLR, ExponentialLR |
| Training Runner | `mynn/runner.py` | Training loop with early stopping |
| Visualization | `visualization.py` | Confusion matrix, weights, kernels, misclassified |

## Results

| Model | Parameters | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| MLP [784, 600, 10] | 477,010 | 93.71% | 93.82% |
| MLP + Momentum | 477,010 | 95.75% | 95.40% |
| MLP + L2 Weight Decay | 477,010 | 95.72% | 95.65% |
| **CNN** | **9,098** | **94.80%** | **95.79%** |

CNN achieves higher accuracy with **52x fewer parameters** than MLP.

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Train MLP baseline
python run_experiments.py mlp

# Train CNN
python run_experiments.py cnn

# Run all optimization experiments
python run_experiments.py opt

# Run all regularization experiments
python run_experiments.py reg

# Evaluate on test set
python evaluate_all.py

# Generate visualizations
python visualization.py
```

## Key Implementation Details

- **im2col convolution**: Sliding window patches extracted and reshaped for matrix multiplication, achieving 105x speedup over nested loops
- **He initialization**: `W ~ N(0, sqrt(2/fan_in))` for ReLU networks — using default `std=1` caused CNN to start at 11.5% (random); He init jumped to 81.3% on first evaluation
- **Combined softmax-cross-entropy gradient**: `d(logits) = softmax(logits) - one_hot(labels)`, avoiding numerical instability of separate softmax Jacobian

## Model Weights

Trained model checkpoints available at: [ModelScope / GitHub Releases link]

## References

- Goodfellow, Bengio, Courville — *Deep Learning* (Chapters 6, 7, 8, 9)
- He et al. — *Delving Deep into Rectifiers* (ICCV 2015)
