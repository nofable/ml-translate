import torch
from ml_translate.transformer import scaled_dot_product_attention


class TestTranspose:
    def test_transpose(self):
        input = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        torch.testing.assert_close(
            input.T, torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8]])
        )

    def test_transpose_with_matmul(self):
        input = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        torch.testing.assert_close(input @ input.T, torch.tensor([[30, 70], [70, 174]]))
        torch.testing.assert_close(
            torch.matmul(input, input.T), torch.tensor([[30, 70], [70, 174]])
        )


class TestDivide:
    def test_divide(self):
        input = torch.tensor([4, 16, 36, 48])
        torch.testing.assert_close(input / 2, torch.tensor([2.0, 8.0, 18.0, 24.0]))

    def test_sqrt(self):
        input = torch.tensor([4, 16, 36, 48])
        torch.testing.assert_close(
            input / torch.sqrt(torch.tensor([4])), torch.tensor([2.0, 8.0, 18.0, 24.0])
        )


class TestScaledDotProductAttention:
    def test_attention(self):
        q = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        k = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        v = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
        output = scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(output, torch.tensor([[3.0, 3.0], [3.0, 3.0]]))

    def test_identical_queries_produce_identical_rows(self):
        q_row = torch.normal(1, 0.5, size=(1, 2))
        q = torch.stack([q_row, q_row])
        k = torch.normal(1, 0.5, size=(2, 2))
        v = torch.normal(2, 0.25, size=(2, 2))
        output = scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(output[0], output[1])

    def test_output_sums(self):
        q = torch.normal(1, 0.5, size=(2, 3))
        k = torch.normal(1, 0.5, size=(2, 3))
        v = torch.normal(2, 0.25, size=(2, 3))
        output = scaled_dot_product_attention(q, k, v)
        v_min = torch.min(v)
        v_max = torch.max(v)
        for _, row in enumerate(output):
            i_min = torch.min(row)
            i_max = torch.max(row)
            assert v_min <= i_min
            assert i_max <= v_max

    def test_matching_key_gets_highest_weight(self):
        q = torch.tensor([[0.0, 5.0], [5.0, 5.0]])
        k = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        v = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        output = scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(output, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))

    def test_mask(self):
        mask = torch.tril(torch.ones((2, 2)))
        q = torch.tensor([[0.0, 5.0], [5.0, 5.0]])
        k = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        v = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        output = scaled_dot_product_attention(q, k, v, mask=mask)
        torch.testing.assert_close(output, torch.tensor([[1.0, 0.0], [0.5, 0.5]]))
