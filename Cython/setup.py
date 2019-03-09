from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize('RMSE_cy.pyx'), include_dirs=[numpy.get_include()])
# python setup.py build_ext --inplace
# for html file > cython -a RMSE_cy.pyx
