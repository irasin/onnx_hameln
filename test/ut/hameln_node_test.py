import pytest

import onnx
import onnx.helper as onnx_helper
from onnx_hameln import HamelnNode, HamelnTensor


class TestHamelnNode:

    def setup_method(self):

        self.node = HamelnNode(
            onnx_helper.make_node(op_type="Conv",
                                  inputs=["X", "W", "B"],
                                  outputs=["Y"],
                                  name="conv_node",
                                  stride=[1, 1]))

    def test_non_node(self):
        with pytest.raises(ValueError) as e:
            _ = HamelnNode(1)
        assert "node should be NodeProto" in str(e.value)

    def test_get_op_type(self):
        assert self.node.get_op_type() == "Conv"

    def test_get_name(self):
        assert self.node.get_name() == "conv_node"

    def test_get_attribute(self):
        attrs = onnx_helper.make_attribute(key="stride", value=[1, 1])
        assert len(self.node.get_attribute()) == 1
        assert self.node.get_attribute()[0] == attrs

    def test_add_input(self):
        # if we construct hameln_node from onnx_model
        # self.node.input will contain the hameln_tensors of X/W/B
        assert self.node.input == []
        assert self.node.node.input == ["X", "W", "B"]
        ht = HamelnTensor(onnx.ValueInfoProto(name="another_tensor"))

        self.node.add_input(ht)

        assert self.node.input == [ht]
        assert self.node.node.input == ["X", "W", "B", "another_tensor"]
        with pytest.raises(ValueError) as e:
            self.node.add_input(1)
        assert "input should be HamelnTensor" in str(e.value)

    def test_connect(self):
        import copy

        next_node = copy.copy(self.node)

        assert self.node.to_node == []
        assert next_node.from_node == []

        HamelnNode.connect(self.node, next_node)

        assert self.node.to_node == [next_node]
        assert next_node.from_node == [self.node]

        another_node = copy.copy(self.node)

        HamelnNode.connect(self.node, another_node)
        assert self.node.to_node == [next_node, another_node]

        HamelnNode.connect(self.node, another_node, clear_before_connect=True)
        assert self.node.to_node == [another_node]

    def test_eq_hash(self):
        import copy

        c1 = copy.copy(self.node)
        c2 = copy.deepcopy(self.node)
        c3 = self.node

        assert c1 == self.node
        assert c2 == self.node
        assert c3 == self.node

        assert c1 is not self.node
        assert c2 is not self.node
        assert c3 is self.node

        assert hash(c1) != hash(self.node)
        assert hash(c2) != hash(self.node)
        assert hash(c3) == hash(self.node)
