# ONNX model rewriter tool


a pure python tool to rewrite and optimize onnx model


## usage

```python
from onnx_hameln import HamelnModel, HPM


m = HamelnModel("yolov5l_v3.onnx")

HPM.rewrite(m)

m.set_batch_size(32).set_nhwc_input_format().export("rewrite.onnx")


```
