import polars as pl
from labels import surely_pos, surely_neg, maybe_neg

df = pl.scan_parquet('./data/all_features.parquet').with_row_index('row_idx')
total = df.select(pl.count()).collect().item()

for name, indices in [
    ("surely_pos", surely_pos),
    ("surely_neg", surely_neg),
    ("maybe_neg", maybe_neg),
]:
    found = df.filter(pl.col('row_idx').is_in(indices)).select(pl.count()).collect().item()
    print(f"{name:12s}: {len(indices):5d} total, {found:5d} found in parquet")

print(f"\nParquet rows: {total}")
