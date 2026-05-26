from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True

    def __call__(self, X):
        return self.forward(X)

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        dX = grad @ self.W.T
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer with im2col-based vectorization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.b = initialize_method(size=(out_channels,))
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def _im2col(self, X_pad):
        N, C_in, H_pad, W_pad = X_pad.shape
        kH, kW = self.kernel_size
        stride = self.stride
        H_out = (H_pad - kH) // stride + 1
        W_out = (W_pad - kW) // stride + 1

        cols = np.zeros((N, C_in, kH, kW, H_out, W_out))
        for h in range(kH):
            for w in range(kW):
                cols[:, :, h, w, :, :] = X_pad[:, :,
                    h:h + H_out * stride:stride,
                    w:w + W_out * stride:stride]
        cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
        return cols, H_out, W_out

    def _col2im(self, cols, X_pad_shape):
        N, C_in, H_pad, W_pad = X_pad_shape
        kH, kW = self.kernel_size
        stride = self.stride
        H_out = (H_pad - kH) // stride + 1
        W_out = (W_pad - kW) // stride + 1

        cols = cols.reshape(N, H_out, W_out, C_in, kH, kW).transpose(0, 3, 4, 5, 1, 2)
        dX_pad = np.zeros(X_pad_shape)
        for h in range(kH):
            for w in range(kW):
                # Use np.add.at for overlapping accumulation
                sl_h = slice(h, h + H_out * stride, stride)
                sl_w = slice(w, w + W_out * stride, stride)
                dX_pad[:, :, sl_h, sl_w] += cols[:, :, h, w, :, :]
        return dX_pad

    def forward(self, X):
        self.input = X
        self.X_pad_shape = None
        N, C_in, H, W = X.shape
        kH, kW = self.kernel_size
        pad, stride = self.padding, self.stride

        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            X_pad = X

        H_out = (H + 2 * pad - kH) // stride + 1
        W_out = (W + 2 * pad - kW) // stride + 1
        self.X_pad_shape = X_pad.shape

        cols, _, _ = self._im2col(X_pad)
        W_col = self.W.reshape(self.out_channels, -1)
        out = cols @ W_col.T + self.b
        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grads):
        X = self.input
        N, C_in, H, W = X.shape
        kH, kW = self.kernel_size
        pad = self.padding

        if pad > 0:
            X_pad = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
        else:
            X_pad = X

        H_out, W_out = grads.shape[2], grads.shape[3]

        cols, _, _ = self._im2col(X_pad)

        grad_col = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        dW_col = cols.T @ grad_col
        dW = dW_col.T.reshape(self.out_channels, C_in, kH, kW)

        db = np.sum(grads, axis=(0, 2, 3))

        W_col = self.W.reshape(self.out_channels, -1)
        dX_col = grad_col @ W_col

        dX_pad = self._col2im(dX_col, X_pad.shape)

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db

        if pad > 0:
            return dX_pad[:, :, pad:pad + H, pad:pad + W]
        else:
            return dX_pad

    def clear_grad(self):
        self.grads = {'W': None, 'b': None}


class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride=None) -> None:
        super().__init__()
        self.optimizable = False
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]
        self.input = None
        self.max_positions = None

    def forward(self, X):
        self.input = X
        N, C, H, W = X.shape
        kH, kW = self.kernel_size
        stride = self.stride
        H_out = (H - kH) // stride + 1
        W_out = (W - kW) // stride + 1

        output = np.zeros((N, C, H_out, W_out))
        self.max_positions = np.zeros((N, C, H_out, W_out, 2), dtype=int)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        patch = X[n, c, h * stride:h * stride + kH, w * stride:w * stride + kW]
                        max_val = np.max(patch)
                        argmax = np.unravel_index(np.argmax(patch), patch.shape)
                        output[n, c, h, w] = max_val
                        self.max_positions[n, c, h, w] = [
                            h * stride + argmax[0],
                            w * stride + argmax[1]
                        ]
        return output

    def backward(self, grads):
        X = self.input
        N, C, H, W = X.shape
        dX = np.zeros_like(X)
        for n in range(N):
            for c in range(C):
                for h in range(grads.shape[2]):
                    for w in range(grads.shape[3]):
                        h_pos, w_pos = self.max_positions[n, c, h, w]
                        dX[n, c, h_pos, w_pos] += grads[n, c, h, w]
        return dX


class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)


class Dropout(Layer):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        self.optimizable = False
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, X):
        if self.training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1.0 - self.p)
            return X * self.mask
        return X

    def backward(self, grads):
        if self.training:
            return grads * self.mask
        return grads


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.optimizable = False
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.logits = None
        self.labels = None
        self.softmax_out = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.logits = predicts
        self.labels = labels
        self.softmax_out = softmax(predicts)
        N = predicts.shape[0]
        loss = -np.mean(np.log(
            self.softmax_out[np.arange(N), labels] + 1e-10
        ))
        return loss

    def backward(self):
        N = self.logits.shape[0]
        grad = self.softmax_out.copy()
        grad[np.arange(N), self.labels] -= 1.0
        grad /= N
        self.model.backward(grad)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition