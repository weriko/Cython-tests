from distutils.core import setup, Extension
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.docstrings = False
import numpy
import hashlib

NN = Extension('NeuralNetwork', ['NeuralNetwork.pyx'], include_dirs=[numpy.get_include()])
primes = Extension('FindPrimes', ['FindPrimes.pyx'], include_dirs=[numpy.get_include()])
pow = Extension('POW', ['POW.pyx'], include_dirs=[numpy.get_include()])
exts = (cythonize([NN, primes, pow],
                  compiler_directives={'language_level': "3", 
				       'boundscheck':False}))

setup(ext_modules=exts)