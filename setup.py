from onnx_hameln import __version__
from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name="onnx_hameln",
      version=__version__,
      author="irasin",
      author_email="edad811@gmail.com",
      url="https://github.com/irasin/onnx_hameln",
      description="an onnx rewrite tool",
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=["numpy", "onnx", "onnxoptimizer", "networkx"],
      python_requires='>=3.6',
      license="Apache License 2.0")
