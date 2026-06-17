def get_pandas_view_of_polars_df(df):
    """Convert Polars DataFrame to Pandas view (copy if necessary)."""
    return df.to_pandas()
