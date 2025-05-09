"""
Utility functions (lag creation, metric name cleaning, etc.)
"""

"""
Utility functions
"""

import re

def clean_metric_name(name: str) -> str:
    """
    Sanitize station names into MLflow‑safe metric names.
    """
    # allow alphanumerics, underscores, hyphens, periods, colons, slashes, spaces
    return re.sub(r"[^\w\-\.:/ ]", "_", name)

