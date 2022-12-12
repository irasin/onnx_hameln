import pytest
import copy

import onnx

from onnx import TensorProto
import onnx.checker as onnx_checker
import onnx.helper as onnx_helper
from onnx_hameln import HamelnGraph, HamelnNode, HamelnTensor


class TestHamelnNode:

    def setup_method(self):

        X = onnx_helper.make_tensor_value_info('X', TensorProto.FLOAT,
                                               [3, 3, 2, 2])
        pads = onnx_helper.make_tensor('pads', TensorProto.FLOAT, [4],
                                       [1, 1, 4, 4])
        value = onnx_helper.make_tensor('value', TensorProto.FLOAT, [1], [1])
        Y = onnx_helper.make_tensor_value_info('Y', TensorProto.FLOAT,
                                               [3, 3, 4, 4])

        # Create a node (NodeProto) - This is based on Pad-11
        node_def = onnx_helper.make_node(
            op_type='Pad',  # node name
            inputs=['X', 'pads', 'value'],  # inputs
            outputs=['Y'],  # outputs
            name="pad_node",
            mode='constant',  # attributes
        )

        # Create the graph (GraphProto)
        graph_def = onnx_helper.make_graph(nodes=[node_def],
                                           name="test_model",
                                           inputs=[X],
                                           outputs=[Y],
                                           initializer=[pads, value])

        onnx_checker.check_graph(graph_def)

        self.graph = HamelnGraph(graph_def)
        self.graph.node = [HamelnNode(node_def)]
        self.graph.tensor = [HamelnTensor(X), HamelnTensor(Y)]
        self.graph.weight = [HamelnTensor(pads), HamelnTensor(value)]
        self.graph.graph_input = [HamelnTensor(X)]
        self.graph.graph_output = [HamelnTensor(Y)]

        self.node = HamelnNode(node_def)

    def test_non_graph(self):
        with pytest.raises(ValueError) as e:
            _ = HamelnGraph(1)
        assert "graph should be GraphProto" in str(e.value)

    def test_get_name(self):
        assert self.graph.get_name() == "test_model"

    def test_set_name(self):
        self.graph.set_name("new_name")
        assert self.graph.get_name() == "new_name"

    def test_count_op_type(self):

        cnt = self.graph.count_op_type()
        assert len(cnt) == 1
        assert cnt["Pad"] == 1

    def test_get_all_node_name(self):
        assert self.graph.get_all_node_name() == ["pad_node"]

    def test_get_batch_size(self):
        assert self.graph.get_batch_size() == 3

    def test_set_batch_size(self):
        self.graph.set_batch_size(4)
        assert self.graph.get_batch_size() == 4

    def test_set_nhwc_input_format(self):
        assert self.graph.graph_input[0].get_dim() == [3, 3, 2, 2]
        self.graph.set_nhwc_input_format()
        assert self.graph.graph_input[0].get_dim() == [3, 2, 2, 3]

    def test_get_node_by_op_type(self):
        assert len(self.graph.get_node_by_op_type("Pad")) == 1
        assert len(self.graph.get_node_by_op_type("Conv")) == 0

    def test_get_node_by_name(self):
        assert len(self.graph.get_node_by_name("pad_node")) == 1

        with pytest.raises(ValueError) as e:
            self.graph.get_node_by_name("conv_node")
            assert "can not find node with name" in str(e.value)

    def test_get_index_by_node(self):

        assert self.graph.get_index_of_node(self.node) == 0

        c1 = copy.copy(self.graph)
        with pytest.raises(Exception):
            self.graph.get_index_of_node(c1)

        c2 = HamelnNode(onnx.NodeProto())
        with pytest.raises(Exception):
            self.graph.get_index_of_node(c2)

    def test_get_tensor_by_name(self):
        for name in ["X", "Y", "pads", "value"]:
            assert self.graph.get_tensor_by_name(name) is not None

        with pytest.raises(ValueError) as e:
            self.graph.get_tensor_by_name("1")
            assert "can not find tensor with name" in str(e.value)

    def test_topological_sort_hameln_node(self):
        # TODO: add more test case
        assert HamelnGraph.topological_sort_hameln_node(
            self.graph.node) is not None

    def test_add_internal_tensor_to_graph_output(self):
        # TODO: add more test case
        with pytest.raises(Exception):
            # failed because it only has one node
            self.graph.add_internal_tensor_to_graph_output()

    def test_extract_subgraph(self):
        # TODO: add more test case
        with pytest.raises(Exception):
            self.graph.extract_subgraph([], [])

    def test_eq_hash(self):

        c1 = copy.copy(self.graph)
        c2 = copy.deepcopy(self.graph)
        c3 = self.graph

        assert c1 == self.graph
        assert c2 == self.graph
        assert c3 == self.graph

        assert c1 is not self.graph
        assert c2 is not self.graph
        assert c3 is self.graph

        assert hash(c1) != hash(self.graph)
        assert hash(c2) != hash(self.graph)
        assert hash(c3) == hash(self.graph)
