def is_jupyter_notebook():
    """Detect if running inside a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is None:
            return False
        return "IPKernelApp" in get_ipython().config
    except Exception:
        return False
