import os
# Avoid automatic extra parallelization by other modules
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
os.environ['NUMEXPR_NUM_THREADS'] = "1"

import click
from joblib import dump
import logging
from pathlib import Path

from astroML.density_estimation import XDGMM

from proc.pipeline import load_and_preproc


N_COMPONENTS = 5
MAX_ITER = 150


@click.command()
@click.option("-f", help="Input fits file", required=True)
@click.option("-bmin", type=float, help="Minimum galactic latitude", required=True)
@click.option("-bmax", type=float, help="Maximum galactic latitude", required=True)
@click.option("-lmin", type=float, help="Minimim galactic longitude", required=True)
@click.option("-lmax", type=float, help="Maximum galactic longitude", required=True)
@click.option("-d", help="Output directory to save processed files", required=True)
@click.option("-sample", type=int, help="Sample size", required=False)
@click.option("-idname", help="Name", required=False)
def main(f, bmin, bmax, lmin, lmax, d, sample=None, idname=""):
    f = Path(f)  
    # Check input file exists
    if not f.exists():
        raise FileNotFoundError(f"{f} not found")

    d = Path(d)
    # Check output directory exists and create it if not
    if not d.exists():
        d.mkdir(parents=True)
    elif not d.is_dir():
        raise NotADirectoryError(f"{d} is not a directory")

    # Check consistency of input parameters
    if bmin >= bmax:
        raise ValueError(f"{bmin} >= {bmax}")
    elif lmin >= lmax:
        raise ValueError(f"{lmin} >= {lmax}")

    # Initialize logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=d / f"logfile_{idname}.txt",
        level=logging.INFO,
    )
    logging.info("Started")
    logging.info("Input file: %s", f)
    logging.info("Output directory: %s", d)
    logging.info("Galactic latitude: %s %s", bmin, bmax)
    logging.info("Galactic longitude: %s %s", lmin, lmax)
    logging.info("Sample size: %s", sample)
    logging.info("Name: %s", idname)
    logging.info("Number of components: %s", N_COMPONENTS)
    logging.info("Maximum number of iterations: %s", MAX_ITER)

    # Load data
    logging.info("Loading data...")
    df, data = load_and_preproc(f, bmin, bmax, lmin, lmax, sample)
    

    # Perform
    logging.info("Fitting XDGMM...")
    xdgmm = XDGMM(n_components=N_COMPONENTS, max_iter=MAX_ITER)
    xdgmm.fit(data["astrom"], data["astrom_cov"])

    logging.info(f"Means: \n{xdgmm.mu}")
    logging.info(
        f"Covariances: \n{xdgmm.V}",
    )
    logging.info(
        f"Weights: \n{xdgmm.alpha}",
    )

    # Save
    logging.info("Saving XDGMM model...")
    dump(xdgmm, d / f"xgmm_{idname}.joblib")


if __name__ == "__main__":
    main()
