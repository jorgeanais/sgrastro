import numpy as np
import pandas as pd
from pathlib import Path

from functools import partial, reduce
from typing import Callable, Optional
import numpy.typing as npt

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

def load(filepath: Path) -> pd.DataFrame:
    """
    Read the data from the input file.
    It is expected a csv file with the columns names
    or a fits table
    """

    suffix = filepath.suffix

    if suffix == ".fits":
        from astropy.table import Table

        table = Table.read(filepath, format="fits")
        df = table.to_pandas()

    elif suffix == ".csv":
        df = pd.read_csv(filepath)

    else:
        raise ValueError(f"File extension {suffix} not recognized")

    return df

def filter_df_columns(df: pd.DataFrame, cols_to_keep: list[str]) -> pd.DataFrame:
    """
    Keep only specified columns.
    """
    df = df.copy()
    return df[cols_to_keep]

def drop_rows_with_nan_values(df: pd.DataFrame, cols_to_check: list[str]) -> pd.DataFrame:
    """
    Remove rows with NaN values in the specified columns.
    """
    df = df.copy()
    return df.dropna(subset=cols_to_check)

def ruwe_criterion(df: pd.DataFrame, threshold: float = 1.3) -> pd.DataFrame:
    df = df.copy()
    return df.query("ruwe < @threshold")

def latitude_filter(df: pd.DataFrame, bmin: Optional[float] = None, bmax: Optional[float] = None) -> pd.DataFrame:
    if bmin is None:
        bmin = -np.inf
    if bmax is None:
        bmax = np.inf
    return df.query("b > @bmin and b < @bmax")

def longitude_filter(df: pd.DataFrame, lmin: Optional[float] = None, lmax: Optional[float] = None) -> pd.DataFrame:
    if lmin is None:
        lmin = -np.inf
    if lmax is None:
        lmax = np.inf
    return df.query("l > @lmin and l < @lmax")

def add_sgr_pm_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Correct proper motions based on Vasiliev et al. 2020"""

    df = df.copy()

    SGR_RA = 283.764
    SGR_DEC = -30.480
    SGR_MU_RA = -2.692
    SGR_MU_DEC = -1.359

    ra_1 = df["ra"].to_numpy()
    dec_1 = df["dec"].to_numpy()

    delta_ra = ra_1 - SGR_RA
    delta_dec = dec_1 - SGR_DEC

    new_mu_alpha = (
        -2.69
        + 0.009 * delta_ra
        - 0.002 * delta_dec
        - 0.00002 * delta_ra * delta_ra * delta_ra
    )
    new_mu_delta = (
        -1.35
        - 0.024 * delta_ra
        - 0.019 * delta_dec
        - 0.00002 * delta_ra * delta_ra * delta_ra
    )

    df["pmra_corr"] = df["pmra"] - new_mu_alpha
    df["pmdec_corr"] = df["pmdec"] - new_mu_delta

    return df

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

    return output

def create_colors(
    df: pd.DataFrame, colors: list[str] = ["mag_J-mag_Ks"]
) -> pd.DataFrame:
    """
    Create colors from the magnitudes. The colors are added as new columns to the dataframe.
    """

    df = df.copy()

    for color in colors:
        band1, band2 = color.split("-")
        df[color] = df[f"{band1}"] - df[f"{band2}"]

    return df

def get_correlation_matrix(dx: npt.NDArray, dy: npt.NDArray, X: npt.NDArray):
    """
    Construct the correlation matrix for the proper motions.

    Parameters
    ----------
    dx : array_like
        The proper motion error in the ra direction.
    dy : array_like
        The proper motion error in the dec direction.
    X : array_like
        The proper motions.
    """
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([dx ** 2, dy ** 2]).T
    return Xerr

def get_correlation_matrix_3x3(dx: npt.NDArray, dy: npt.NDArray, dz: npt.NDArray, X: npt.NDArray):
    """
    Construct the correlation matrix for the proper motions.

    Parameters
    ----------
    dx : array_like
        The proper motion error in the ra direction.
    dy : array_like
        The proper motion error in the dec direction.
    dz : array_like
        The parallax error.
    X : array_like
        The proper motions.
    """
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([dx ** 2, dy ** 2, dz ** 2]).T
    return Xerr


def get_arrays(
    filepath: Path,
    sample_size: Optional[int] = None,
    bmin: Optional[float] = None,
    bmax: Optional[float] = None,
    lmin: Optional[float] = None,
    lmax: Optional[float] = None,
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

    std_pipeline = [
        partial(latitude_filter, bmin=bmin, bmax=bmax),
        partial(longitude_filter, lmin=lmin, lmax=lmax),
        partial(filter_df_columns, cols_to_keep=COLS_TO_KEEP),
        partial(drop_rows_with_nan_values, cols_to_check=COLS_TO_KEEP),
        add_sgr_pm_correction,
    ]

    print(f"Perfoming standard preprocessing {filepath.name}")
    df = load(filepath)
    preprocess = _compose(*std_pipeline)
    df = preprocess(df)

    
    if sample_size is not None:
        sample_size = min(sample_size, len(df))
        df = df.sample(sample_size)

    # Correlation proper motions
    outdict = get_subspace_data(df, SPACE_PARAMS)
    outdict["astrom_cov"] = get_correlation_matrix_3x3(outdict["astrom_err"][:,0], outdict["astrom_err"][:,1], outdict["astrom_err"][:,2], outdict["astrom"])
    outdict["astrom_corr_cov"] = get_correlation_matrix_3x3(outdict["astrom_err"][:,0], outdict["astrom_err"][:,1], outdict["astrom_err"][:,2], outdict["astrom_corr"])

    return df, outdict