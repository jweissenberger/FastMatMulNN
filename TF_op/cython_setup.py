from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize

ext_module = Extension(
    "openmpext",
    ["cython_openmp_ext.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name='OpenMP app',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module]
)