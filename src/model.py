import warnings
from bisect import bisect_right
from collections import Counter

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, _LRScheduler
from torchvision.models import resnet50


class ImageNetClassifier(pl.LightningModule):
    def __init__(
        self,
        finetune,
        data_fraction,
        pretrained_chkpt,
        temperature,
        logits_file,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.finetune = finetune
        self.data_fraction = data_fraction

        resnet = resnet50(pretrained=False)
        if pretrained_chkpt:
            state_dict = torch.load(pretrained_chkpt)["state_dict"]
            resnet.load_state_dict(state_dict, strict=False)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Linear(2048, 1000)

        self.temperature = temperature

        self.logits_file = logits_file

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x

    def _loss(self, x, y):
        # kl div
        if self.logits_file:
            log_probs = F.log_softmax(x, dim=1)
            y_probs = F.softmax(y / self.temperature, dim=1)
            loss = F.kl_div(log_probs, y_probs, reduction="batchmean")
        # cross entropy
        else:
            loss = F.cross_entropy(x, y)

        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        x = self.backbone(x)
        x = x.flatten(1)
        x = self.classifier(x)

        loss = self._loss(x, y)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        x = self.backbone(x)
        x = x.flatten(1)
        x = self.classifier(x)

        if len(y.shape) == 2:
            y = y.argmax(1)

        return {"true": y, "logits": x}

    def test_epoch_end(self, test_outs):
        truths = torch.cat([out["true"] for out in test_outs])
        logits = torch.cat([out["logits"] for out in test_outs])

        acc1, acc5 = accuracy(logits, truths, (1, 5))

        print(f"acc1: {acc1.item():.2f}")
        print(f"acc5: {acc5.item():.2f}")

    def configure_optimizers(self):
        # https://arxiv.org/pdf/2005.04966.pdf
        if self.finetune:
            print(f"Finetuning for {self.data_fraction} of the data.")
            if self.data_fraction == 0.01:
                classifier_lr = 1.0
            elif self.data_fraction == 0.1:
                classifier_lr = 0.1
            else:
                classifier_lr = 0.01

            optimizer = torch.optim.SGD(
                [
                    {"params": self.backbone.parameters()},
                    {"params": self.classifier.parameters(), "lr": classifier_lr},
                ],
                lr=0.01,
                momentum=0.9,
                weight_decay=5e-4,
            )
            scheduler = MultiStepLR(optimizer, [12, 16], 0.2)
        # https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhai_S4L_Self-Supervised_Semi-Supervised_Learning_ICCV_2019_paper.pdf
        else:
            if self.data_fraction == 0.01:
                wd = 1e-2
                lr = 0.01
            elif self.data_fraction == 0.1:
                wd = 1e-3
                lr = 0.1
            else:
                wd = 1e-4
                lr = 0.1

            optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, momentum=0.9, weight_decay=wd
            )

            if self.data_fraction == 0.01:
                scheduler = S4LScheduler(optimizer, [700, 800, 900], 10, 0.1)
            elif self.data_fraction == 0.1:
                scheduler = S4LScheduler(optimizer, [140, 160, 180], 5, 0.1)
            else:
                scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# https://pytorch.org/docs/1.5.0/_modules/torch/optim/lr_scheduler.html#MultiStepLR
class S4LScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.
    Added linear warmup.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, warmup_until, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup_until = warmup_until
        self.linear_growth = None
        super(S4LScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch < self.warmup_until:
            if not self.linear_growth:
                self.linear_growth = [
                    group["lr"] / self.warmup_until
                    for group in self.optimizer.param_groups
                ]
                return self.linear_growth
            else:
                return [
                    group["lr"] + growth
                    for group, growth in zip(
                        self.optimizer.param_groups, self.linear_growth
                    )
                ]
        else:
            if self.last_epoch not in self.milestones:
                return [group["lr"] for group in self.optimizer.param_groups]
            return [
                group["lr"] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups
            ]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_until:
            return [
                base_lr * self.last_epoch / self.warmup_until
                for base_lr in self.base_lrs
            ]
        else:
            milestones = list(sorted(self.milestones.elements()))
            return [
                base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]


# https://github.com/facebookresearch/moco/blob/main/main_moco.py
def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
