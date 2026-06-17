import logging

logger = logging.getLogger(__name__)


def plot_model_feature_importances(
    model=None,
    columns=None,
    model_name="",
    num_factors=20,
    show_plots=False,
    plot_file=None,
    **kwargs,
):
    """Minimal stub: feature importance plotting is skipped."""
    if plot_file:
        logger.debug(f"Feature importance plot skipped: {plot_file}")
