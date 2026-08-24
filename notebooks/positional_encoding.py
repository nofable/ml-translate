# %% [markdown]
# Positional Encoding

# %%
import torch
import pandas as pd
import altair as alt
from ml_translate.model import PositionalEncoder

pe = PositionalEncoder(20, 100)
output = pe.forward(torch.zeros(size=(100, 20)))
data = pd.concat(
    [
        pd.DataFrame(
            {
                "embedding": output[:, dim],
                "dimension": dim,
                "position": list(range(100)),
            }
        )
        for dim in [5, 6, 7, 8]
    ]
)

alt.Chart(data).mark_line().properties(width=800).encode(
    x="position", y="embedding", color="dimension:N"
).interactive()
# %%
