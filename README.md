# Welcome to the *PyVoroTomo* (version 2!) repository!

## >> this is very much beta code-- use at your own risk! bug reports appreciated<<

This is a revamped version of the original PyVoroTomo by Fang et al. (2020). This code implements a Poisson Voronoi-based seismic traveltime tomography method, primarily suited for regional or smaller study areas but may also be expanded to do joint teleseismic inversions. See changelog for details. PyKonal (White et al., 2020) is used for calculating traveltimes and tracing rays.

![Raypaths](figures/logo.png)

## Documentation
# Installation
Easiest to clone into your directory: 
`$ git pull https://github.com/filefolder/PyVoroTomo.git`
and install locally via pip, `$ python -m pip install ./PyVoroTomo`
... which will install the **pyvorotomo** executable

NOTE! Version 2 also requires PyKonal (>= 0.5.0) which is only available here: https://github.com/filefolder/pykonal_0.5.git
(Primarily this incorporates a critical NumPy update, but there are a few other hacks and enhancements)

See also: [https://github.com/malcolmw/PyVoroTomo/wiki/Installation](https://github.com/malcolmw/PyVoroTomo/wiki/Installation)

# Running
The code runs best with mpiexec / openmpi. Maybe someday it will run mp natively, but not any time soon. 

`$ mpiexec -n 16 pyvorotomo -r -c experiment.cfg`

parameters:

-n number of CPU<br>
-v enable verbose/debug logging<br>
-r relocate catalog first<br>
-t ONLY run the sensitivity test<br>
-x output all realizations<br>

-c /path/to/configfile.cfg<br>


(note that stations, events, output/log paths, etc are now listed in the config file, NOT as commandline arguments)

The output log can be quite long, but key status updates are flagged with a '###' one can grep for (e.g. `grep '###' pyvorotomo.log`)


# Config File
See example.cfg for available options, explanations and recommendations!

## Code Structure

**bin**: Where the python executable **pyvorotomo** lives<br>
**pyvorotomo**: Module code. The main one you probably want to look at is _iterator.py<br>
**scripts**: Some tools to generate stations.h5, catalog.h5, and other common tasks (work in progress)<br>
**synth_data**: Some synthetic data to experiment with<br>
**real_data**: Some example real-world data<br>


## Citation
If you make use of this code in published work, please cite the below (more very soon I expect)

## Installation (previous version)
Refer to [https://github.com/malcolmw/PyVoroTomo/wiki/Installation](https://github.com/malcolmw/PyVoroTomo/wiki/Installation) for both laptops and clusters.

## References
1. Fang, H., van der Hilst, R. D., de Hoop, M. V., Kothari, K., Gupta, S., & Dokmanić, I. (2020). Parsimonious seismic tomography with Poisson Voronoi projections: Methodology and validation. Seismological Research Letters, 91(1), 343-355.
2. White, M. C. A., Fang, H., Nakata, N., & Ben-Zion, Y. (2020). PyKonal: A Python Package for Solving the Eikonal Equation in Spherical and Cartesian Coordinates Using the Fast Marching Method. *Seismological Research Letters, 91*(4), 2378-2389. https://doi.org/10.1785/0220190318
