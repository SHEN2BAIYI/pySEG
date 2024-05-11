import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=0.001, num_classes=1, weights=None):
        super(DiceLoss, self).__init__()
        self.__dict__.update(locals())

    def forward(self, logits, targets):
        # binary
        if self.num_classes == 1:
            preds = logits.contiguous().view(logits.size(0), -1).sigmoid()
            targets = targets.contiguous().view(targets.size(0), -1).float()
            loss = self.compute_loss(preds, targets)
            return loss

        # multi-class
        else:
            preds = logits.softmax(axis=1).contiugous().view(logits.size(0), self.num_classes, -1)
            t = targets.contiguous().view(targets.size(0), -1)
            targets = torch.nn.functional.one_hot(t, self.num_classes).permute(0, 2, 1)
            total_loss = 0
            for i in range(self.num_classes):
                dice_loss = self.compute_loss(preds[:, i], targets[:, i])
                if self.weights is not None:
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
            return total_loss

    def compute_loss(self, preds, targets):
        a = torch.sum(preds * targets, dim=1)
        b = torch.sum(preds * preds, dim=1) + self.smooth
        c = torch.sum(targets * targets, dim=1) + self.smooth
        d = (2 * a) / (b + c)
        return torch.mean(1 - d)


class MixedLoss(nn.Module):
    def __init__(self, bce_ratio=0.5):
        super(MixedLoss, self).__init__()

        self.bce_ratio = bce_ratio
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_ratio * bce_loss + (1 - self.bce_ratio) * dice_loss
