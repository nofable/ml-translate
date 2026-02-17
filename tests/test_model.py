import numpy as np
from ml_translate.model import (
    AddAndNorm,
    FeedForward,
    softmax,
    ReLU,
    scaled_dot_product_attention,
)


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


class TestReLU:
    def test_relu_all_positive(self):
        x = np.random.uniform(low=-2, high=2, size=(10, 10))
        output = ReLU(x)
        assert np.all(output >= 0)

    def test_relu_positive_passthru(self):
        x = np.random.uniform(low=0, high=2, size=(10, 10))
        output = ReLU(x)
        np.testing.assert_array_equal(x, output)


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        n_positions = 10
        x = np.random.uniform(low=-2, high=-2, size=(n_positions, 5))
        output = softmax(x)
        sums = np.sum(output, axis=-1)
        np.testing.assert_almost_equal(sums, np.ones(n_positions))

    def test_softmax_all_positive(self):
        x = np.random.uniform(low=-4, high=-2, size=(10, 10))
        output = softmax(x)
        assert np.all(output > 0)

    def test_softmax_max_is_highest_prob(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        x_arg_max = np.argmax(x, axis=-1)
        output = softmax(x)
        arg_max = np.argmax(output, axis=-1)
        np.testing.assert_equal(x_arg_max, arg_max)


class TestScaledDotProductAttention:
    def test_attention(self):
        q = np.array([[0, 1], [0, 1]])
        k = np.array([[1, 1], [1, 1]])
        v = np.array([[2, 2], [4, 4]])
        output = scaled_dot_product_attention(q, k, v)
        np.testing.assert_equal(output, np.array([[3, 3], [3, 3]]))

    def test_identical_queries_produce_identical_rows(self):
        q_row = np.random.normal(1, 0.5, size=2)
        q = np.stack([q_row, q_row])
        k = np.random.normal(1, 0.5, size=(2, 2))
        v = np.random.normal(2, 0.25, size=(2, 2))
        output = scaled_dot_product_attention(q, k, v)
        np.testing.assert_equal(output[0], output[1])

    def test_output_sums(self):
        q = np.random.normal(1, 0.5, size=(2, 3))
        k = np.random.normal(1, 0.5, size=(2, 3))
        v = np.random.normal(2, 0.25, size=(2, 3))
        output = scaled_dot_product_attention(q, k, v)
        v_min = np.min(v)
        v_max = np.max(v)
        for _, row in enumerate(output):
            i_min = np.min(row)
            i_max = np.max(row)
            assert v_min <= i_min
            assert i_max <= v_max

    def test_matching_key_gets_highest_weight(self):
        q = np.array([[0, 5], [5, 5]])
        k = np.array([[0, 1], [1, 0]])
        v = np.array([[2, 2], [2, 2]])
        output = scaled_dot_product_attention(q, k, v)
        np.testing.assert_equal(output, np.array([[2, 2], [2, 2]]))

    def test_mask(self):
        mask = np.tril(np.ones((2, 2)))
        q = np.array([[0, 5], [5, 5]])
        k = np.array([[0, 1], [1, 0]])
        v = np.array([[1, 0], [0, 1]])
        output = scaled_dot_product_attention(q, k, v, mask=mask)
        np.testing.assert_equal(output, np.array([[1, 0], [0.5, 0.5]]))
