import pytest

import numpy as np
from onnx import ValueInfoProto
import onnx.helper as onnx_helper
from onnx_hameln import HamelnTensor


class TestHamelnTensor:

    def setup_method(self):
        self.value_info_tensor = HamelnTensor(
            onnx_helper.make_tensor_value_info(name="value_info",
                                               elem_type=1,
                                               shape=[32, 3, 224, 224]))

        self.value_info_name_only_tensor = HamelnTensor(
            ValueInfoProto(name="value_info_name_only"))

        self.weight_tensor = HamelnTensor(
            onnx_helper.make_tensor(name="weight_tensor",
                                    data_type=1,
                                    dims=[2, 3, 4],
                                    vals=np.arange(2 * 3 * 4,
                                                   dtype="f").reshape(2, 3,
                                                                      4)))

    def test_non_tensor(self):
        with pytest.raises(Exception) as e:
            _ = HamelnTensor(1)
        assert "tensor should be TensorProto or ValueInfoProto" in str(e.value)

    def test_get_node(self):
        assert self.value_info_tensor.get_node() is None

    def test_set_node(self):

        dummy_node_object = "dummy_node_object"
        self.value_info_tensor.set_node(dummy_node_object)

        assert self.value_info_tensor.get_node() == dummy_node_object

    def test_get_name(self):
        assert self.value_info_tensor.get_name() == "value_info"

    def test_set_name(self):

        self.value_info_tensor.set_name("value_info_new")
        assert self.value_info_tensor.get_name() == "value_info_new"

    def test_get_dim(self):
        assert self.value_info_tensor.get_dim() == [32, 3, 224, 224]
        assert self.value_info_name_only_tensor.get_dim() == []
        assert self.weight_tensor.get_dim() == [2, 3, 4]

    def test_set_dim(self):
        with pytest.raises(ValueError) as e:
            self.weight_tensor.set_dim([3, 4])
            assert "can not set dim to weight tensor, use set_data instead" in str(
                e.value)

        self.value_info_tensor.set_dim([1, 2, 3, 4])
        assert self.value_info_tensor.get_dim() == [1, 2, 3, 4]

        with pytest.raises(ValueError) as e:
            self.value_info_name_only_tensor.set_dim([1, 2, 3, 4])
            assert "can not set dim to empty value_info" in str(e.value)

    def test_get_data(self):

        assert self.weight_tensor.get_data() is not None

        with pytest.raises(ValueError) as e:
            self.value_info_tensor.get_data()
            assert "can not get data from value_info" in str(e.value)

        with pytest.raises(ValueError) as e:
            self.value_info_name_only_tensor.get_data()
            assert "can not get data from value_info" in str(e.value)

    def test_set_data(self):

        data = np.arange(3 * 2 * 1 * 4).reshape(3, 2, 1, 4)
        self.weight_tensor.set_data(data)
        assert np.all(self.weight_tensor.get_data() == data)
        assert self.weight_tensor.get_dim() == [3, 2, 1, 4]

        with pytest.raises(ValueError) as e:
            self.value_info_tensor.set_data(data)
            assert "can not set data to value_info" in str(e.value)

        with pytest.raises(ValueError) as e:
            self.value_info_name_only_tensor.set_data(data)
            assert "can not set data to value_info" in str(e.value)

    def test_set_batch_size(self):
        with pytest.raises(ValueError):
            self.weight_tensor.set_batch_size(10)

        self.value_info_tensor.set_batch_size(10)
        assert self.value_info_tensor.get_dim()[0] == 10

        with pytest.raises(ValueError):
            self.value_info_name_only_tensor.set_batch_size(10)

    def test_nchw_dim_to_nhwc_dim(self):
        nchw = [1, 2, 3, 4]
        nhwc = [1, 3, 4, 2]

        assert HamelnTensor.nchw_dim_to_nhwc_dim(nchw) == nhwc

        with pytest.raises(Exception):
            HamelnTensor.nchw_dim_to_nhwc_dim([1, 2, 3])

    def test_has_data(self):
        assert self.value_info_tensor.has_data is False
        assert self.value_info_name_only_tensor.has_data is False
        assert self.weight_tensor.has_data is True

    def test_eq_hash(self):
        import copy

        c1 = copy.copy(self.value_info_tensor)
        c2 = copy.deepcopy(self.value_info_tensor)
        c3 = self.value_info_tensor

        assert c1 == self.value_info_tensor
        assert c2 == self.value_info_tensor
        assert c3 == self.value_info_tensor

        assert c1 is not self.value_info_tensor
        assert c2 is not self.value_info_tensor
        assert c3 is self.value_info_tensor

        assert hash(c1) != hash(self.value_info_tensor)
        assert hash(c2) != hash(self.value_info_tensor)
        assert hash(c3) == hash(self.value_info_tensor)
