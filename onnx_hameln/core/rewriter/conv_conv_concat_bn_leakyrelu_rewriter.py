import numpy as np

import onnx
import onnx.helper as onnx_helper

from ..pattern import HamelnPattern
from ..hameln import HamelnGraph, HamelnNode, HamelnTensor

conv_conv_concat_bn_leakyrelu = HamelnPattern(
).construct_pattern_from_definition(
    op_type_list=["Conv", "Conv", "Concat", "BatchNormalization", "LeakyRelu"],
    linkage=[[0, 2], [1, 2], [2, 3], [3, 4]])


def _conv_conv_concat_bn_leakyrelu_rewrite(hameln_graph: HamelnGraph,
                                           node_mapping):
    conv_0: HamelnNode = hameln_graph.node[node_mapping[0]]
    conv_1: HamelnNode = hameln_graph.node[node_mapping[1]]
    concat_2: HamelnNode = hameln_graph.node[node_mapping[2]]
    batchnormalization_3: HamelnNode = hameln_graph.node[node_mapping[3]]
    leaky_relu_4: HamelnNode = hameln_graph.node[node_mapping[4]]
    """
        conv:
        y1 = W @ input + B ( @ stands for conv )

        bn:
        y2 = ((y1 - mean) / sqrt(var)) * weight + bias

        ==》

        fused_conv_bn
        y2 = weight/sqrt(var) * y1 + bias -  mean * weight/sqrt(var)
        = weight/sqrt(var) * W  @ input + (B - mean) * weight / sqrt(var) + bias


        let scale = weight/sqrt(var)
        then,
            new_W = scale * W
            new_B = scale * (B - mean) + bias

    """

    conv_0_input = conv_0.input
    conv_0_weight = conv_0_input[1]
    conv_0_weight_data = conv_0_weight.get_data()
    if len(conv_0_input) == 3:
        conv_0_bias = conv_0_input[2]
    else:
        conv_0_bias = onnx_helper.make_tensor(
            name=conv_0_weight.get_name().replace("weight", "bias") +
            "_hameln",
            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
                np.float32)],
            dims=[conv_0_weight_data.shape[0]],
            vals=np.zeros((conv_0_weight_data.shape[0]), dtype=np.float32))
        conv_0_bias = HamelnTensor(conv_0_bias)
        conv_0.add_input(conv_0_bias)
        hameln_graph.weight.append(conv_0_bias)
    conv_0_bias_data = conv_0_bias.get_data()

    conv_1_input = conv_1.input
    conv_1_weight = conv_1_input[1]
    conv_1_weight_data = conv_1_weight.get_data()
    if len(conv_1_input) == 3:
        conv_1_bias = conv_1_input[2]
    else:
        conv_1_bias = onnx_helper.make_tensor(
            name=conv_1_weight.get_name().replace("weight", "bias") +
            "_hameln",
            data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
                np.float32)],
            dims=[conv_1_weight_data.shape[0]],
            vals=np.zeros((conv_1_weight_data.shape[0]), dtype=np.float32))
        conv_1_bias = HamelnTensor(conv_1_bias)
        conv_1.add_input(conv_1_bias)
        hameln_graph.weight.append(conv_1_bias)
    conv_1_bias_data = conv_1_bias.get_data()

    bn_3_input = batchnormalization_3.input
    bn_3_weight_data = bn_3_input[1].get_data()
    bn_3_bias_data = bn_3_input[2].get_data()
    bn_3_mean_data = bn_3_input[3].get_data()
    bn_3_var_data = bn_3_input[4].get_data()

    bn_weight_data1, bn_weight_data2 = np.split(bn_3_weight_data, 2)
    bn_bias_data1, bn_bias_data2 = np.split(bn_3_bias_data, 2)
    bn_mean_data1, bn_mean_data2 = np.split(bn_3_mean_data, 2)
    bn_var_data1, bn_var_data2 = np.split(bn_3_var_data, 2)

    scale1 = bn_weight_data1 / np.sqrt(bn_var_data1)
    conv_0_weight_data = conv_0_weight_data * scale1.reshape(-1, 1, 1, 1)
    conv_0_bias_data = (conv_0_bias_data -
                        bn_mean_data1) * scale1 + bn_bias_data1

    scale2 = bn_weight_data2 / np.sqrt(bn_var_data2)
    conv_1_weight_data = conv_1_weight_data * scale2.reshape(-1, 1, 1, 1)
    conv_1_bias_data = (conv_1_bias_data -
                        bn_mean_data2) * scale2 + bn_bias_data2

    conv_0_weight.set_data(conv_0_weight_data)
    conv_0_bias.set_data(conv_0_bias_data)
    conv_1_weight.set_data(conv_1_weight_data)
    conv_1_bias.set_data(conv_1_bias_data)

    left_leaky_relu_output_name = conv_0.output[0].get_name(
    ) + "_leaky_relu_output"

    left_leaky_relu = HamelnNode(
        onnx_helper.make_node(op_type=leaky_relu_4.get_op_type(),
                              inputs=[conv_0.output[0].get_name()],
                              outputs=[left_leaky_relu_output_name],
                              name=conv_0.get_name() + "_leaky_relu",
                              alpha=leaky_relu_4.get_attribute()[0].f))

    right_leaky_relu_output_name = conv_1.output[0].get_name(
    ) + "leaky_relu_output"

    right_leaky_relu = HamelnNode(
        onnx_helper.make_node(op_type=leaky_relu_4.get_op_type(),
                              inputs=[conv_1.output[0].get_name()],
                              outputs=[right_leaky_relu_output_name],
                              name=conv_1.get_name() + "_leaky_relu",
                              alpha=leaky_relu_4.get_attribute()[0].f))

    concat_output_name = leaky_relu_4.output[0].get_name()

    last_concat = HamelnNode(
        onnx_helper.make_node(
            op_type=concat_2.get_op_type(),
            inputs=[left_leaky_relu_output_name, right_leaky_relu_output_name],
            outputs=[concat_output_name],
            name=concat_2.get_name(),
            axis=1))

    # reconnect subgraph
    HamelnNode.connect(conv_0, left_leaky_relu, clear_before_connect=True)
    HamelnNode.connect(conv_1, right_leaky_relu, clear_before_connect=True)
    HamelnNode.connect(left_leaky_relu, last_concat)
    HamelnNode.connect(right_leaky_relu, last_concat)
    HamelnNode.connect(last_concat,
                       leaky_relu_4.to_node[0],
                       clear_before_connect=True)

    remove_node = [concat_2, batchnormalization_3, leaky_relu_4]
    insert_node = [left_leaky_relu, right_leaky_relu, last_concat]

    return True, remove_node, insert_node


conv_conv_concat_bn_leakyrelu.register_rewriter(
    _conv_conv_concat_bn_leakyrelu_rewrite)
