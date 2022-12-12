# ONNX model rewriter tool


a pure python tool to rewrite and optimize onnx model

[![build_test_upload](https://github.com/irasin/onnx_hameln/actions/workflows/python-package.yml/badge.svg)](https://github.com/irasin/onnx_hameln/actions/workflows/python-package.yml)

## usage

```python
from onnx_hameln import HamelnModel, HPM


m = HamelnModel("yolov5l_v3.onnx")

HPM.rewrite(m)

m.set_batch_size(32).set_nhwc_input_format().export("rewrite.onnx")


```
