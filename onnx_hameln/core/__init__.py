from .hameln import HamelnModel, HamelnGraph, HamelnNode, HamelnTensor

from .pattern import HamelnPatternManager

from .rewriter.conv_conv_concat_bn_leakyrelu_rewriter import conv_conv_concat_bn_leakyrelu

HPM = HamelnPatternManager()

HPM.register_pattern(pattern_name="conv_conv_concat_bn_leakyrelu",
                     pattern=conv_conv_concat_bn_leakyrelu)
