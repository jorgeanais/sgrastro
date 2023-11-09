# sgrastro

This program is used to study the proper motions of the Sgr dwarf spheroidal galaxy using XDGMM


## How to install

Just install the requirements  (tested on python 3.11.5)
```python
pip install -r requirements.txt
```

If you want to use a virtual environment, you can create it with
```
python -m venv venv
```
and then activate it with
```
source venv/bin/activate
```
After that install the requirements as indicated before.

# Usage

```bash
python main.py  -f <file> \
                -bmin <float> \
                -bmax <float> \
                -lmin <float> \
                -lmax <float> \
                -d <dir> \
                -sample <int> \
                -idname <name> \
```

Example:
```bash
python main.py  -f /home/jorge/Documents/data/vvvxtiles/raw/Gaia/gaia_primary_sample_region_low_bulge.fits \
                -bmin -15.0 \
                -bmax -8.0 \
                -lmin 4.0 \
                -lmax 9.0 \
                -d /home/jorge/test01 \
                -sample 1000 \
                -idname test01 \
```