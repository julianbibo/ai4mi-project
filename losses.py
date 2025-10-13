#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
from torch import einsum
import torch

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']

        # class balancing weights
        if kwargs["weighted"]:
            self.weights = torch.tensor([
                9.09798426e-04, 1.88867803e+00, 1.16511524e-01, 2.54913241e+00, 4.44768241e-01
            ])
        else:
            self.weights = None

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        # weight loss
        if self.weights is not None:
            mask = mask * self.weights.to(mask.device).view(1, -1, 1, 1)

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10


        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)

class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target, smooth=1e-8):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        preds = (pred_softmax[:, self.idk, ...]) # shape torch.Size([8, 5, 256, 256]) (batch, c, h,w)
        mask = weak_target[:, self.idk, ...].float()

        num_classes = preds.shape[1]

        dice = 0

        for c in range(num_classes):  # Loop through each class
            pred_c = preds[:, c, :, :]  # Predictions for class c
            target_c = mask[:, c, :, :]  # Ground truth for class c
            
            intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels
            
            dice += (2 * intersection + smooth) / (union + smooth)  # Per-class Dice score

        return 1 - torch.mean(dice) / num_classes  # Average Dice Loss across batch and classes

class CombinedLoss():
    def __init__(self, idk, ce_weight=0.5, dice_weight=0.5):
        self.ce_loss = CrossEntropy(**idk)
        self.dice_loss = DiceLoss(**idk)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        print(f"Initialized CombinedLoss with ce_weight={ce_weight}, dice_weight={dice_weight}")

    def __call__(self, pred_softmax, weak_target):
        ce = self.ce_loss(pred_softmax, weak_target)
        dice = self.dice_loss(pred_softmax, weak_target)
        total = self.ce_weight * ce + self.dice_weight * dice

        return total


