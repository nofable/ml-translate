from math import nan
import torch
from torch.nn import CrossEntropyLoss


class TestCrossEntropy:
    def test_cross_entropy_zero_loss(self):
        loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        input = torch.tensor([100.0, 0.0, 0.0])
        target = torch.tensor([100.0, 0.0, 0.0])
        loss = loss_fn(input, target)
        assert loss == 0.0

    def test_cross_entropy_max_loss(self):
        loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        input = torch.tensor([100.0, 0.0])
        target = torch.tensor([0.0, 1.0])
        loss = loss_fn(input, target)
        assert loss == 100.0

    def test_cross_entropy_loss_class_indices(self):
        loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss()
        input = torch.tensor([[20.0, 0.0]])
        target = torch.tensor([0])
        loss = loss_fn(input, target)
        assert loss == 0.0

    def test_cross_entropy_ignore_index(self):
        loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss(ignore_index=1)
        input = torch.tensor([[100.0, 0.0], [10.0, 10.0]])
        target = torch.tensor([0, 1])
        loss = loss_fn(input, target)
        assert loss == 0.0

    def test_cross_entropy_label_smoothing(self):
        loss_fn: CrossEntropyLoss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        input = torch.tensor([[100.0, 0.0]])
        target = torch.tensor([0])
        loss = loss_fn(input, target)
        assert loss > 0.0
