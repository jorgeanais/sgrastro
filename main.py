import click
from pathlib import Path

from proc.pre import get_arrays


@click.command()
@click.option("-f", help="Input fits file", required=True)
@click.option("-bmin", type=float, help="Minimum galactic latitude", required=True)
@click.option("-bmax", type=float, help="Maximum galactic latitude", required=True)
@click.option("-lmin", type=float, help="Minimim galactic longitude", required=True)
@click.option("-lmax", type=float, help="Maximum galactic longitude", required=True)
@click.option("-d", help="Output directory to save processed files", required=True)
@click.option("-sample", type=float, help="Sample size", required=False)
def main(f, bmin, bmax, lmin, lmax, d, sample=None):
    
    f = Path(f)
    # Check input file exists
    if not f.exists():
        raise FileNotFoundError(f"{f} not found")
    
    d = Path(d)
    # Check output directory exists
    if not d.exists():
        raise FileNotFoundError(f"{d} not found")
    
    # Check consistency of input parameters
    if bmin >= bmax:
        raise ValueError(f"{bmin} >= {bmax}")
    elif lmin >= lmax:
        raise ValueError(f"{lmin} >= {lmax}")
    
    # Load data
    df, data = get_arrays(f, sample, bmin, bmax, lmin, lmax)

    
    


if __name__ == '__main__':
    main()