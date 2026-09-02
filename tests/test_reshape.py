import torch
from torch import Tensor


class TestReshape:
    def test_reshape(self):
        batch: Tensor = torch.rand((10, 30, 3000))
        one_dim = batch.reshape(-1)
        assert one_dim.shape == (900000,)
        two_dim = batch.reshape(-1, batch.size(-1))
        assert two_dim.shape == (300, 3000)
