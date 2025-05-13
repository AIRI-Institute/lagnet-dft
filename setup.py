# MIT License
#
# Copyright (c) 2025 AIRI - Artificial Intelligence Research Institute
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import pathlib
from setuptools import find_packages, setup

def get_python_version():
    version_info = sys.version_info
    major = version_info[0]
    minor = version_info[1]
    return f"cp{major}{minor}"
PYTHON_VERSION = get_python_version()


setup(
    name="LAGNet-DFT",
    version="0.0.23",
    author="Konstantin Ushenin",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=["numpy",
                    "wandb",
                    "e3nn",
                    "scikit-learn",
                    "hruid",
                    "tensorboard",
                    "pytorch_lightning",
                    "hydra-core",
                    "ase",
                    "pyscf",
                    "pandas",
                    "matplotlib",
                    "seaborn"],
    license="MIT",
    description="LAGNet, invariant and equivariant DeepDFT",
    long_description="""LAGNet-DFT is an implementation of LAGNet, equivariant DeepDFT, and invariant DeepDFT. This implementation uses the adjacent matrix and mask-aggregation.""",
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)