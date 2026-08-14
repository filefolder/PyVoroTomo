import io
import numpy as np
import os
from setuptools import setup

name            = "pyvorotomo"
description     = "Version 2 of a Parsimonious Voronoi-cell based tomography code (originally by Fang et al., 2019)"
url             = "https://github.com/filefolder/PyVoroTomo"
email           = "robert.pickle@anu.edu.au"
author          = "Robert Pickle and Hongjian Fang and Malcolm C. A. White"
requires_python = ">=3.10"
packages        = ["pyvorotomo"]
required        = [
    "KDEpy>=1.1.12",
    "mpi4py>=4.1.2",
    "numpy>=2.4.0",
    "pandas",
    "pykonal>=0.5.5",
    "tables",
    "scipy>=1.16.0"
]
scripts         = ["bin/pyvorotomo"]
license         = "GNU GPLv3"

here = os.path.abspath(os.path.dirname(__file__))

about = {}
project_slug = name.lower().replace("-", "_").replace(" ", "_")
with open(os.path.join(here, project_slug, "__version__.py")) as f:
    exec(f.read(), about)

setup(
    name=name,
    version=about["__version__"],
    description=description,
    author=author,
    author_email=email,
    python_requires=requires_python,
    url=url,
    packages=packages,
    scripts=scripts,
    install_requires=required,
    license=license,
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: 3.15",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Physics"
    ]
)
