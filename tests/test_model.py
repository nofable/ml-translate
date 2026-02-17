import numpy as np
from ml_translate.model import AddAndNorm, FeedForward


class TestAddAndNorm:
    def test_add_and_norm_zeros(self):
        addNorm = AddAndNorm(shape=4)
        x = np.zeros((4, 4))
        sublayer_output = np.ones((4, 4))
        output = addNorm.forward(x, sublayer_output)
        np.testing.assert_equal(output, np.zeros((4, 4)))

    def test_add_and_norm_ones(self):
        addNorm = AddAndNorm(shape=2, eps=0)
        x = np.array([[2, 4], [4, 2]])
        sublayer_output = np.ones((2, 2))
        output = addNorm.forward(x, sublayer_output)
        np.testing.assert_equal(output, [[-1, 1], [1, -1]])


class TestFeedForward:
    def test_feed_forward(self):
        x = [[-3, 1], [2, 2]]
        ff = FeedForward(d_model=2, d_hidden=4)
        output = ff.forward(x)
        np.testing.assert_equal(output, [[0.0, 0.0], [16.0, 16.0]])
