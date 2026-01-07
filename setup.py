"""
Setup script for causalfe with optional C++ extension.
"""

from setuptools import setup, find_packages

# Try to build C++ extension, fall back to pure Python
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "causalfe.cpp.cffe_core",
            ["causalfe/cpp/cffe_core.cpp"],
            cxx_std=14,
            extra_compile_args=["-O3"],
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    ext_modules = []
    cmdclass = {}
    print("pybind11 not found, building pure Python version")

setup(
    name="causalfe",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
    ],
    extras_require={
        "dev": ["pytest", "pybind11"],
        "compare": ["econml", "scikit-learn"],
    },
)
