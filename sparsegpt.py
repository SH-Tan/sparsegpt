import math
import time

import torch
import torch.nn as nn

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


import math
import time
import torch
import torch.nn as nn

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.dev = layer.weight.device

        W = layer.weight.data
        self.is_conv = isinstance(layer, nn.Conv2d)

        if self.is_conv:
            W = W.flatten(1)   # [out, in*k*k]

        self.rows, self.columns = W.shape

        self.H = torch.zeros((self.columns, self.columns),
                             device=self.dev,
                             dtype=torch.float32)

        self.nsamples = 0

    # ------------------------------------------------
    # Collect Hessian (input correlation)
    # ------------------------------------------------
    def add_batch(self, inp, out):

        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        if inp.ndim == 2:
            inp = inp.unsqueeze(0)

        if self.is_conv:
            # unfold conv input → im2col
            unfold = nn.Unfold(
                kernel_size=self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)                     # [B, C*k*k, L]
            inp = inp.permute(1, 0, 2).reshape(self.columns, -1)

        elif isinstance(self.layer, nn.Linear):
            if inp.ndim == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t()

        tmp = inp.shape[1]

        # running average (numerically stable)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp @ inp.t()

    # ------------------------------------------------
    # Fast prune
    # ------------------------------------------------
    def fasterprune(self,
                    sparsity,
                    blocksize=128,
                    percdamp=0.01):

        W = self.layer.weight.data
        orig_dtype = W.dtype

        if self.is_conv:
            W = W.flatten(1)

        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        
        # dead columns
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # damping
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp

        # inverse Hessian via Cholesky
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        losses = torch.zeros(self.rows, device=self.dev)

        for i1 in range(0, self.columns, blocksize):

            i2 = min(i1 + blocksize, self.columns)

            W1 = W[:, i1:i2].clone()
            Hinv1 = Hinv[i1:i2, i1:i2]

            # importance
            # eps = 1e-12
            # denom = torch.diag(Hinv1).view(1, -1)

            # scores = W1**2 / (denom**2 + eps)
            scores = W1**2 / (torch.diag(Hinv1).view(1, -1)**2)

            thresh = torch.quantile(scores.flatten(), sparsity)
            mask = scores <= thresh
            
            # print(torch.max(scores), torch.min(scores), thresh)

            Q1 = W1.clone()
            Q1[mask] = 0

            err = (W1 - Q1) / torch.diag(Hinv1).view(1, -1)

            W[:, i2:] -= err @ Hinv[i1:i2, i2:]
            W[:, i1:i2] = Q1

            losses += torch.sum((W1 - Q1) ** 2, dim=1) / 2

        torch.cuda.synchronize()

        # print(f"time {time.time() - tick:.2f}")
        # print("error", losses.sum().item())

        W = W.reshape(self.layer.weight.shape).to(orig_dtype)
        self.layer.weight.data = W

    def free(self):
        self.H = None
        torch.cuda.empty_cache()
