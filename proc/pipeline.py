import pandas as pd
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from functools import partial, reduce

from proc.pre import (
    load,
    latitude_filter,
    longitude_filter,
    select_columns,
    drop_rows_with_nan_values,
    add_sgr_pm_correction,
    sample,
    get_correlation_matrix,
    get_correlation_matrix_3x3,
)


"""
In this file all the preprocessing functions are patched together.
"""

COLS_TO_KEEP = [
    "ra",
    "dec",
    "l",
    "b",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "parallax",
    "parallax_error",
]

SPACE_PARAMS = {
    "pos_αδ": ["ra", "dec"],
    "pos_lb": ["l", "b"],
    "pm": ["pmra", "pmdec"],
    "pm_corr": ["pmra_corr", "pmdec_corr"],
    "pm_err": ["pmra_error", "pmdec_error"],
    "astrom": ["pmra", "pmdec", "parallax"],
    "astrom_corr": ["pmra_corr", "pmdec_corr", "parallax"],
    "astrom_err": ["pmra_error", "pmdec_error", "parallax_error"],
}


def get_subspace_data(
    df: pd.DataFrame,
    space_params: dict[str, str],
) -> dict[str, npt.NDArray[np.float64]]:
    """
    Return an object which stores the data of each respective sub-space.
    """
    output = dict()
    for key, value in space_params.items():
        output[key] = df[value].to_numpy()

    # Add 2D proper motions
    output["pm_cov"] = get_correlation_matrix(
        output["pm_err"][:, 0], output["pm_err"][:, 1], output["pm"]
    )

    # Add 3D astrometry
    output["astrom_cov"] = get_correlation_matrix_3x3(
        output["astrom_err"][:, 0],
        output["astrom_err"][:, 1],
        output["astrom_err"][:, 2],
        output["astrom"],
    )
    output["astrom_corr_cov"] = get_correlation_matrix_3x3(
        output["astrom_err"][:, 0],
        output["astrom_err"][:, 1],
        output["astrom_err"][:, 2],
        output["astrom_corr"],
    )

    return output


def load_and_preproc(
    filepath: Path,
    bmin: Optional[float] = None,
    bmax: Optional[float] = None,
    lmin: Optional[float] = None,
    lmax: Optional[float] = None,
    sample_size: Optional[int] = None,
) -> tuple[pd.DataFrame, dict[str, npt.NDArray[np.float64]]]:
    """
    Read the data from the input file and return an object which stores the data of each
    respective sub-space.
    """

    PreprocFunction = Callable[[pd.DataFrame], pd.DataFrame]

    def _compose(*functions: PreprocFunction) -> PreprocFunction:
        """
        A function composer, that is, it returns a function which is the composition
        of the input functions.
        Example:
        ``(f o g o h)(x) = f(g(h(x))``
        where f, g and h are functions, and x is the input.
        """
        return reduce(lambda f, g: lambda x: g(f(x)), functions)

    preproc_pipeline = [
        partial(latitude_filter, bmin=bmin, bmax=bmax),
        partial(longitude_filter, lmin=lmin, lmax=lmax),
        partial(select_columns, cols_to_keep=COLS_TO_KEEP),
        partial(drop_rows_with_nan_values, cols_to_check=COLS_TO_KEEP),
        add_sgr_pm_correction,
        partial(sample, sample_size=sample_size),
    ]

    df = load(filepath)
    preprocess = _compose(*preproc_pipeline)
    df = preprocess(df)

    # Generate a dictionary with numpy arrays ready to be processed
    outdict = get_subspace_data(df, SPACE_PARAMS)

    return df, outdict
