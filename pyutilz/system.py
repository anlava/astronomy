import gc

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def tqdmu(iterable, *args, **kwargs):
    """Minimal tqdm wrapper fallback."""
    disable = kwargs.get("disable", False)
    if disable:
        return iterable
    return tqdm(iterable, *args, **kwargs)


def clean_ram():
    """Force garbage collection."""
    gc.collect()
