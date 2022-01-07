### linker

This repository contains software designed for linking single-night transients found in the Dark Energy Survey data into solar system orbits. The algorithms are described in [Bernardinelli et al 2020 ApJS 247 32](https://iopscience.iop.org/article/10.3847/1538-4365/ab6bd8) and [Bernardinelli et al 2021 2109.03758](https://arxiv.org/abs/2109.03758). This branch contains modifications for use in the DECam Ecliptic Exploration Project (DEEP).

The Python software presented here also relies heavily on the C++ package [orbitspp](https://github.com/gbernstein/orbitspp). After installation, an environment variable `ORBITSPP` should point to this software.



#### Usage
The main script, `linker_driver.py`, requires a configuration file (see `examples/example.ini`), which points to the required catalogs as well as defines the search parameters. Two environment variables, `SUBSPLIT` (0 or 1) and `CHUNKSIZE` can change the behavior of the triplet finder step (to split the number of pairs checked at any given time, reducing RAM usage).  


Given a tuple (`RA_0`, `Dec_0`, `t_0`) and an inverse distance bin `gamma_0`, with width `delta gamma`, the process is as follows:
- Exposure list determination (uses `orbitspp`)
- Catalog is reduced to include only data from the determined exposures, and necessary coordinate transformations are computed
- Pairs are found, using kD trees. This step produces two files, one `.npy` and one `.hdf5` file. If these two are present, this step is skipped. These files can be safely deleted once the triplets are found
- Triplets are constructed. This is the most computationally intensive step, and a series of `.npy` files are saved for each exposure/chunk. If these files are present for any given exposure, it is skipped. 
- A final triplet file in fits format is constructed (and so all auxiliary `.npy` and `hdf5` files can be deleted), and is then used for orbit growing, using `orbitspp/GrowOrbits`. Scripts to manage `GrowOrbits` will be added later.


#### Requirements:
```
numpy==1.18.5
skyfield==1.16
astropy==4.0
vaex_jupyter==0.5.2
scipy==1.1.0
numba==0.48.0
vaex==3.0.0
pixmappy==1.0.0
```
