import copy
from collections import defaultdict


import onnx
import onnx.helper as onnx_helper
import onnx.numpy_helper as onnx_numpy_helper

import onnxoptimizer



import networkx as nx
from networkx.algorithms import isomorphism




class HamelnTensor:
    def __init__(self, tensor, is_fake=False):
        self.tensor = tensor
        self.has_data = isinstance(self.tensor, onnx.TensorProto)
        self.node = None
        self.is_fake = is_fake

    def __repr__(self):
        if self.has_data:
            return onnx_helper.printable_tensor_proto(self.tensor)
        else:
            return onnx_helper.printable_value_info(self.tensor)

    def __hash__(self):
        return hash(id(self))
    
    def __eq__(self, other):
        if self.tensor == other.tensor and\
            self.has_data == other.has_data and\
            self.node == other.node and\
            self.is_fake == other.is_fake:
        
            return True
        return False
    
    def __ne__(self, other):
        return not self == other
    
    def set_node(self, node):
        self.node = node

    def get_node(self):
        return self.node

    def get_data(self):
        if self.has_data:
            return onnx_numpy_helper.to_array(self.tensor)
        else:
            raise ValueError(f"can not get data from value_info: {self}")

    def set_data(self, np_data):
        if self.has_data:
            new_tensor = onnx_numpy_helper.from_array(np_data,
                                                      self.tensor.name)
            self.tensor.CopyFrom(new_tensor)
        else:
            raise ValueError(f"can not set data to value_info: {self}")

    def get_name(self):
        return self.tensor.name

    def set_name(self, name):
        self.tensor.name = name

    def get_dim(self):
        if self.has_data:
            return self.tensor.dims
        dims = []
        all_dim = self.tensor.type.tensor_type.shape.dim
        rank = len(all_dim)
        for i in range(rank):
            if all_dim[i].dim_value:
                dims.append(all_dim[i].dim_value)
            else:
                dims.append(all_dim[i].dim_param)
        return dims

    def set_dim(self, dim_value_dict):
        if self.has_data:
            raise ValueError(
                "can not set dim to weight tensor, use set_data instead")
        if isinstance(dim_value_dict, list):
            dim_value_dict = {idx: i for idx, i in enumerate(dim_value_dict)}
        all_dim = self.tensor.type.tensor_type.shape.dim
        for idx, value in dim_value_dict.items():
            if isinstance(value, str):
                all_dim[idx].dim_param = "batch"
            else:
                all_dim[idx].dim_value = value

    def set_batch_size(self, batch_size):
        dim_value_dict = {0: batch_size}
        self.set_dim(dim_value_dict)

    @staticmethod
    def nchw_dim_to_nhwc_dim(dim_list):
        assert len(dim_list) == 4
        new_dim = [dim_list[0], dim_list[2], dim_list[3], dim_list[1]]
        return new_dim
    
    

class HamelnNode:
    def __init__(self, node):
        self.node = node

        self.input = []
        self.output = []

        self.from_node = []
        self.to_node = []

    def __repr__(self):
        return onnx_helper.printable_node(self.node)

    def get_op_type(self):
        return self.node.op_type

    def get_name(self):
        return self.node.name

    def get_attribute(self):
        return self.node.attribute

    def add_input(self, hameln_tensor: HamelnTensor):
        self.node.input.append(hameln_tensor.get_name())
        self.input.append(hameln_tensor)

    @staticmethod
    def connect(from_node, to_node, clear_before_connect=False):
        if clear_before_connect:
            from_node.to_node.clear()
            to_node.from_node.clear()
        from_node.to_node.append(to_node)
        to_node.from_node.append(from_node)


class HamelnGraph:
    def __init__(self, graph):
        self.graph = graph
        self.node = None
        self.tensor = None
        self.weight = None

        self.graph_input = None
        self.graph_output = None

    def __repr__(self):
        return onnx_helper.printable_graph(self.graph)

    def get_name(self):
        return self.graph.name

    def set_name(self, name):
        self.graph.name = name
        return self

    def count_op_type(self):
        assert self.node is not None

        counter = defaultdict(int)
        for i in self.node:
            counter[i.get_op_type()] += 1
        return counter

    def get_all_node_name(self):
        assert self.node is not None
        names = [i.get_name() for i in self.node]
        return names

    def get_batch_size(self):
        assert self.tensor is not None
        return self.tensor[0].get_dim()[0]

    def set_batch_size(self, batch_size):
        if batch_size == self.get_batch_size():
            return self

        # in case that some onnx model inputs contain initializers,
        # we will remove them to avoid rewriting input failure
        tmp_inputs = copy.deepcopy(self.graph_input)
        weight_names = set(i.get_name() for i in self.weight)
        for i in tmp_inputs:
            if i in weight_names:
                self.graph_input.remove(i)

        shape_input = []
        for i in self.node:
            if i.get_op_type() == "Reshape":
                shape_input.extend(i.input)
            if i.get_op_type() == "Resize" and len(i.input) == 4:
                shape_input.append(i.input[3])
        for i in self.weight:
            if i in shape_input:
                shape = i.get_data().copy()  # for write access
                if shape.dtype == "int64":
                    shape[0] = batch_size
                    i.set_data(shape)

        for i in self.tensor:
            try:
                i.set_batch_size(batch_size) # internal tensor may not contain shape info
            except:
                pass
        return self

    def set_nhwc_input_format(self):
        # we can't know whether the original format is nchw or not
        # be careful when use this method

        is_4d_tensor = all(len(i.get_dim()) == 4 for i in self.graph_input)
        if not is_4d_tensor:
            raise ValueError(f"all input tensors must be 4-dimension tensor")

        input_names = [i.get_name() for i in self.graph_input]
        input_shapes = [i.get_dim() for i in self.graph_input]

        nhwc_input_shape = [
            HamelnTensor.nchw_dim_to_nhwc_dim(i) for i in input_shapes
        ]

        for tensor, shape in zip(self.graph_input, nhwc_input_shape):
            tensor.set_dim(shape)

        transpose_nodes_names = [f"{i}_nhwc2nchw" for i in input_names]
        nchw_input_names = [f"{i}_nchw" for i in input_names]

        transpose_node = []
        for i, t, o in zip(input_names, transpose_nodes_names,
                           nchw_input_names):
            new_node = HamelnNode(
                onnx_helper.make_node(op_type="Transpose",
                                      inputs=[i],
                                      outputs=[o],
                                      name=t,
                                      perm=[0, 3, 1, 2]))
            transpose_node.append(new_node)
            self.node.insert(0, new_node)

        for node in self.node:
            for idx, tensor in enumerate(node.input):
                if tensor in self.graph_input:

                    target_name = nchw_input_names[self.graph_input.index(
                        tensor)]
                    target_transpose_node = transpose_node[
                        self.graph_input.index(tensor)]

                    node.node.input[idx] = target_name

                    HamelnNode.connect(target_transpose_node, node)

        self.node = HamelnGraph.topological_sort_hameln_node(self.node)
        return self

    def get_node_by_op_type(self, op_type, with_idx=True):
        idx2node = {}
        for idx, i in enumerate(self.node):
            if i.get_op_type() == op_type:
                idx2node[idx] = i
        if with_idx:
            return idx2node
        node = list(idx2node.values())
        return node

    def get_node_by_name(self, name, with_idx=True):
        idx2node = {}
        for idx, i in enumerate(self.node):
            if i.get_name() == name:
                idx2node[idx] = i

        if len(idx2node) == 0:
            raise ValueError(f"can not find node with name: {name}")
        if with_idx:
            return idx2node
        node = list(idx2node.values())
        return node

    def get_index_of_node(self, node):
        return self.node.index(node)

    def get_tensor_by_name(self, name):
        for i in (self.tensor + self.weight):
            if i.get_name() == name:
                return i

        raise ValueError(f"can not find tensor with name: {name}")

    @staticmethod
    def topological_sort_hameln_node(nodes):
        ## since the original model is a DAG, we do not need to check whether the subgraph is a DAG
        ## TODO(chen.chen): maybe we need add DAG check if we want use this function independently
        V = nodes[:]
        E = []
        for node in V:
            for to_node in node.to_node:
                if to_node in V:
                    E.append((node, to_node))

        def update_input_degree(node_list, edge_list):
            input_degree_map = {}
            for node in node_list:
                if node not in input_degree_map:
                    input_degree_map[node] = 0

                for pair in edge_list:
                    _, to_node = pair
                    if node == to_node:
                        input_degree_map[node] += 1

            return input_degree_map

        def update_zero_candidate(input_degree_map):
            zero_candidate_list = []

            for node, degree in input_degree_map.items():
                if degree == 0:
                    zero_candidate_list.append(node)

            return zero_candidate_list

        def update_graph(node, node_list, edge_list):
            res_node_list = [i for i in node_list if i != node]

            res_edge_list = [pair for pair in edge_list if node not in pair]

            return res_node_list, res_edge_list

        input_degree = update_input_degree(V, E)
        zero_candidate = update_zero_candidate(input_degree)

        topo_order_list = list()

        while len(zero_candidate) != 0:
            top = zero_candidate.pop(0)
            topo_order_list.append(top)

            V, E = update_graph(top, V, E)
            input_degree = update_input_degree(V, E)
            zero_candidate = update_zero_candidate(input_degree)

        return topo_order_list

    def add_internal_tensor_to_graph_output(self, tensor_name=None):
        tensor_list = [
            i for i in self.tensor
            if i not in self.graph_output and i not in self.graph_input
        ]

        if tensor_name is not None:
            tensor_list = [
                i for i in tensor_list if i.get_name() == tensor_name
            ]

        assert len(tensor_list) > 0

        for i in tensor_list:
            self.graph_output.append(i)

        return self

    def extract_subgraph(self, start_nodes, end_nodes):

        assert len(start_nodes), "start nodes are empty"
        assert len(end_nodes), "end nodes are empty"

        subgraph_start_nodes = [
            self.get_node_by_name(i, with_idx=False)[0] for i in start_nodes
        ]
        subgraph_end_nodes = [
            self.get_node_by_name(i, with_idx=False)[0] for i in end_nodes
        ]

        expect_input_tensor = set(i for node in subgraph_start_nodes
                                  for i in node.input if i not in self.weight)
        expect_output_tensor = set(i for node in subgraph_end_nodes
                                   for i in node.output)

        node_stack = subgraph_start_nodes[:]
        subgraph_node = []

        while node_stack:
            top = node_stack.pop(0)
            subgraph_node.append(top)

            to_node = top.to_node
            for i in to_node:
                if i in subgraph_end_nodes:
                    subgraph_node.append(i)
                    continue
                if i in node_stack or i in subgraph_node:
                    continue
                node_stack.append(i)

        actual_input_tensors = set(i for node in subgraph_node
                                   for i in node.input if i not in self.weight)
        actual_output_tensors = set(i for node in subgraph_node
                                    for i in node.output)

        actual_input_tensors, actual_output_tensors = actual_input_tensors - actual_output_tensors, actual_output_tensors - actual_input_tensors

        if actual_input_tensors != expect_input_tensor:
            err_info = f"expected subgraph input are {[i.get_name() for i in expect_input_tensor]},\nactual subgraph input are {[i.get_name() for i in actual_input_tensors]}"

            forgotten_node = [
                i.node.get_name()
                for i in (actual_input_tensors - expect_input_tensor)
            ]
            if forgotten_node:
                err_info = f"{err_info}\n\nmaybe you need add {forgotten_node} into start_nodes"

            redundant_node = [
                i.node.get_name()
                for i in (expect_input_tensor - actual_input_tensors)
            ]
            if redundant_node:
                err_info = f"{err_info}\n\nmaybe you need remove {redundant_node} from start_nodes"

            raise ValueError(err_info)

        if actual_output_tensors != expect_output_tensor:
            err_info = f"expected subgraph output are {[i.get_name() for i in expect_output_tensor]},\nactual subgraph output are {[i.get_name() for i in actual_output_tensors]}, The subgraph does not converge to the specified end_nodes"

            #TODO(chen.chen): add detailed error infomation
            raise ValueError(err_info)

        topo_order_graph = HamelnGraph.topological_sort_hameln_node(
            subgraph_node)
        return topo_order_graph, subgraph_start_nodes, subgraph_end_nodes, list(
            expect_input_tensor), list(expect_output_tensor)


class HamelnModel:
    def __init__(self, model):
        if isinstance(model, str):
            model = onnx.load(model)
        self.parse_onnx(model)
        self.construct_hameln_model()

    @staticmethod
    def add_ms_opset_domain(model,
                            ms_opset_domain="com.microsoft",
                            ms_opset_version=1):
        found = False
        for i in model.opset_import:
            if i.domain == ms_opset_domain:
                found = True
                break

        if not found:
            ms_opset = onnx_helper.make_operatorsetid(ms_opset_domain,
                                                      ms_opset_version)
            model.opset_import.append(ms_opset)

        return model

    @staticmethod
    def preprocess_onnx(model):
        model = HamelnModel.add_ms_opset_domain(model)

        passes = onnxoptimizer.get_available_passes()

        no_need = [
            #TODO(chen.chen): the following passes cause some error, need to debug
            "lift_lexical_references",
            "split_init",
            "split_predict",

            # we do not want to rename anything
            "rename_input_output",
            "set_unique_name_for_nodes"
        ]
        passes = [i for i in passes if i not in no_need]

        model = onnxoptimizer.optimize(model, passes)

        model = onnx.shape_inference.infer_shapes(model,
                                                  check_type=True,
                                                  strict_mode=True,
                                                  data_prop=True)
        onnx.checker.check_model(model)

        return model

    def parse_onnx(self, model):
        model = HamelnModel.preprocess_onnx(model)
        self.model = model
        self.ir_version = self.model.ir_version
        self.opset_import = self.model.opset_import
        self.graph = self.model.graph
        self.node = self.graph.node
        self.input = self.graph.input
        self.output = self.graph.output
        self.initializer = self.graph.initializer
        self.value_info = self.graph.value_info

        return self

    def construct_hameln_model(self):

        self.hameln_node = [HamelnNode(node) for node in self.node]

        self.hameln_input = [HamelnTensor(tensor) for tensor in self.input]
        self.hameln_output = [HamelnTensor(tensor) for tensor in self.output]
        self.hameln_weight = [
            HamelnTensor(tensor) for tensor in self.initializer
        ]
        self.hameln_all_tensor = [
            HamelnTensor(tensor) for tensor in self.value_info
        ] + self.hameln_input + self.hameln_output

        hameln_tensor_weight_map = {
            i.get_name(): i
            for i in self.hameln_all_tensor + self.hameln_weight
        }
        for hameln_node in self.hameln_node:
            node = hameln_node.node
            in_tensor_names = node.input
            out_tensor_names = node.output

            for i in in_tensor_names:
                hameln_node.input.append(hameln_tensor_weight_map[i])

            for o in out_tensor_names:
                if o in hameln_tensor_weight_map:
                    hameln_tensor = hameln_tensor_weight_map[o]
                    if hameln_tensor.get_node() is None:
                        hameln_tensor.set_node(hameln_node)
                        hameln_node.output.append(hameln_tensor)
                else:
                    value_info_proto = onnx.ValueInfoProto()
                    value_info_proto.name = o
                    hameln_tensor = HamelnTensor(value_info_proto)
                    self.hameln_all_tensor.append(hameln_tensor)
                    hameln_tensor_weight_map[o] = hameln_tensor
                    hameln_tensor.set_node(hameln_node)
                    hameln_node.output.append(hameln_tensor)

        for hameln_node in self.hameln_node:
            for i in hameln_node.input:
                if i.get_node():
                    from_node = i.get_node()
                    hameln_node.from_node.append(from_node)
                    from_node.to_node.append(hameln_node)

        self.hameln_graph = HamelnGraph(self.graph)
        self.hameln_graph.node = self.hameln_node
        self.hameln_graph.tensor = self.hameln_all_tensor
        self.hameln_graph.weight = self.hameln_weight
        self.hameln_graph.graph_input = self.hameln_input
        self.hameln_graph.graph_output = self.hameln_output

        return self

    def set_batch_size(self, batch_size):
        self.hameln_graph.set_batch_size(batch_size)
        self.update()
        return self

    def set_nhwc_input_format(self):
        self.hameln_graph.set_nhwc_input_format()
        self.update()
        return self

    def update(self):
        ori_graph = self.hameln_graph.graph
        nodes = [i.node for i in self.hameln_graph.node]
        name = self.hameln_graph.get_name()

        inputs = [i.tensor for i in self.hameln_graph.graph_input]
        outputs = [i.tensor for i in self.hameln_graph.graph_output]
        initializer = [i.tensor for i in self.hameln_graph.weight]
        graph = onnx_helper.make_graph(nodes=nodes,
                                       name=name,
                                       inputs=inputs,
                                       outputs=outputs,
                                       initializer=initializer,
                                       doc_string=ori_graph.doc_string)
        self.model.graph.CopyFrom(graph)
        self.model = HamelnModel.preprocess_onnx(self.model)

    def export(self, save_path=None):
        self.update()

        if save_path:
            onnx.save(self.model, save_path)
