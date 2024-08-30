"Handle build of the Cython code."
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize
import os
from distutils.core import setup


def read(fname):
    "Read a file in the current directory."
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="ahrs_cython",
    author="Romain Fayat",
    version="0.1",
    author_email="r.fayat@gmail.com",
    description="Cython implementation of AHRS filters.",
    ext_modules=cythonize("ahrs_cython/*.pyx"),
    include_dirs=[np.get_include()],
    install_requires=["numpy", "Cython", "ahrs"],
    packages=["madgwick"],
    long_description=read('README.md')
)
