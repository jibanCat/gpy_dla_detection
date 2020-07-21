from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules=[
    Extension("fast_dla_log_model_evidence",
              ["gpy_dla_detection/fast_dla_log_model_evidence.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp" ],
              extra_link_args=['-fopenmp']
    ) 
]

setup( 
  name = "DLA_log_model_evidence",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)

# setup(name="DLA log model evidence", ext_modules=cythonize('gpy_dla_detection/fast_dla_log_model_evidence.pyx'),)
