


#%%
import copy
from collections import defaultdict

import numpy as np
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
        for i in tmp_inputs:
            if i in self.weight:
                self.graph_input.remove(i)

        shape_input = []
        for i in self.node:
            if i.get_op_type() == "Reshape":
                shape_input.extend(i.input)
            if i.get_op_type == "Resize" and len(i.input) == 4:
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


class HamelnPatternNode:
    def __init__(self,
                 idx,
                 op_type=None,
                 from_type=None,
                 to_type=None,
                 from_idx=None,
                 to_idx=None):
        self.idx = idx
        self.op_type = op_type
        self.from_type = from_type if from_type else []
        self.to_type = to_type if to_type else []
        self.from_idx = from_idx if from_idx else []
        self.to_idx = to_idx if to_idx else []

        self.link_cnt = 0

    def __repr__(self):
        return f"idx: {self.idx}: , op_type: {self.op_type}, from_type: {self.from_type}, from_idx: {self.from_idx}, to_type: {self.to_type}, to_idx: {self.to_idx}"

    def get_link_cnt(self):
        link_cnt = self.link_cnt
        self.link_cnt += 1
        return link_cnt


class HamelnPatternTree:
    def __init__(self, pattern_nodes=None):
        self.pattern_nodes = pattern_nodes if pattern_nodes else []

        self.pattern_graph = None

    def __repr__(self):
        return "\n".join([str(i) for i in self.pattern_nodes])

    def complie(self):
        g = nx.DiGraph()
        for i in self.pattern_nodes:
            g.add_node(i.idx,
                       op_type=i.op_type,
                       from_type=i.from_type,
                       to_type=i.to_type,
                       from_idx=i.from_idx,
                       to_idx=i.to_idx)

        for i in self.pattern_nodes:
            for to_idx in i.to_idx:
                link_cnt = self.pattern_nodes[to_idx].get_link_cnt()
                g.add_edge(i.idx, to_idx, link_cnt=link_cnt)
        self.pattern_graph = g
        return self

    def show(self):
        assert self.pattern_graph is not None, f"call compile graph before show"

        nx.draw_kamada_kawai(self.pattern_graph, with_labels=True)
        print(self)


class HamelnPattern:
    def __init__(self):
        self.pattern_tree = None
        self.rewriter = None

    def _construct(self, all_nodes, start_nodes, end_nodes):

        pattern_nodes = [HamelnPatternNode(i) for i in range(len(all_nodes))]

        for idx, i in enumerate(all_nodes):
            pnode = pattern_nodes[idx]
            pnode.op_type = i.get_op_type()

            from_node = i.from_node
            to_node = i.to_node

            if i in start_nodes:
                pnode.to_type = [ii.get_op_type() for ii in to_node]
                pnode.to_idx = [all_nodes.index(ii) for ii in to_node]
            elif i in end_nodes:
                pnode.from_type = [ii.get_op_type() for ii in from_node]
                pnode.from_idx = [all_nodes.index(ii) for ii in from_node]
            else:
                pnode.to_type = [ii.get_op_type() for ii in to_node]
                pnode.to_idx = [all_nodes.index(ii) for ii in to_node]
                pnode.from_type = [ii.get_op_type() for ii in from_node]
                pnode.from_idx = [all_nodes.index(ii) for ii in from_node]

        self.pattern_tree = HamelnPatternTree(pattern_nodes).complie()

        return self

    def construct_pattern_from_subgraph(
        self,
        all_nodes,
        start_nodes=None,
        end_nodes=None,
        inputs=None,
        outputs=None,
    ):
        #TODO(chen.chen): I don't know here what exactly we need from subgraph...
        return self._construct(all_nodes, start_nodes, end_nodes)

    def construct_pattern_from_graph(self, all_nodes):

        start_nodes = [i for i in all_nodes if len(i.from_node) == 0]
        end_nodes = [i for i in all_nodes if len(i.to_node) == 0]

        return self._construct(all_nodes, start_nodes, end_nodes)

    def construct_pattern_from_definition(self, op_type_list, linkage):
        pattern_nodes = [
            HamelnPatternNode(i) for i in range(len(op_type_list))
        ]

        for idx, op_type in enumerate(op_type_list):
            pattern_nodes[idx].op_type = op_type

        for link in linkage:
            from_idx, to_idx = link
            pattern_nodes[from_idx].to_idx.append(to_idx)
            pattern_nodes[from_idx].to_type.append(
                pattern_nodes[to_idx].op_type)
            pattern_nodes[to_idx].from_idx.append(from_idx)
            pattern_nodes[to_idx].from_type.append(
                pattern_nodes[from_idx].op_type)

        self.pattern_tree = HamelnPatternTree(pattern_nodes).complie()

        return self

    def register_rewriter(self, rewriter_func):
        self.rewriter = rewriter_func

    def match(self, hameln_graph):

        complete_graph = HamelnPattern().construct_pattern_from_graph(
            hameln_graph.node)

        def node_match(left, right):
            equal = left["op_type"] == right["op_type"]
            return equal

        def edge_match(left, right):
            equal = left["link_cnt"] == right["link_cnt"]
            return equal

        gm = isomorphism.DiGraphMatcher(
            complete_graph.pattern_tree.pattern_graph,
            self.pattern_tree.pattern_graph, node_match, edge_match)
        gm.subgraph_is_isomorphic()

        matching = list(gm.subgraph_isomorphisms_iter())
        return matching

    def rewrite_graph(self, hameln_graph: HamelnGraph):
        matching = self.match(hameln_graph)

        #TODO(chen.chen): wtf, I don't know why I add a status variable here, remove it
        rewrite_success = True

        remove_nodes = []
        insert_nodes = []
        for mapping in matching:
            inverse_node_mapping = {v: k for k, v in mapping.items()}
            status, remove_node, insert_node = self.rewriter(
                hameln_graph, inverse_node_mapping)
            rewrite_success &= status
            remove_nodes.extend(remove_node)
            insert_nodes.extend(insert_node)

        for i in remove_nodes:
            hameln_graph.node.remove(i)
        hameln_graph.node.extend(insert_nodes)
        hameln_graph.node = HamelnGraph.topological_sort_hameln_node(
            hameln_graph.node)

        return rewrite_success

    def rewrite_model(self, hameln_model: HamelnModel):

        return self.rewrite_graph(hameln_model.hameln_graph)



class HamelnPatternManager:
    _instance = None
    
    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __init__(self):        
        self.all_pattern = {}
        
    
    def __repr__(self):
        info = f"HamelnPatternManager with {len(self.all_pattern)} patterns:\n\n"
        if len(self.all_pattern) > 0:
            info += "\n".join([f"\t{i}" for i in list(self.all_pattern.keys())])
        return info
        
    def get_available_pattern(self):
        return list(self.all_pattern.keys())



    def get_pattern(self, pattern_name):
        pattern = self.all_pattern.get(pattern_name, None)
        if pattern is None:
            raise ValueError(f"can not find pattern: {pattern_name}")
        return pattern
    
    
    def register_pattern(self, pattern_name, pattern):
        if pattern_name in self.all_pattern:
            raise KeyError(f"pattern: {pattern_name} exists")
        self.all_pattern[pattern_name] = pattern


    def rewrite(self, hameln_graph_or_model, pattern_name=None):
        if pattern_name is None:
            pattern_list = self.get_available_pattern()
        elif isinstance(pattern_name, str):
            pattern_list = [pattern_name]
        elif isinstance(pattern_name, list):
            pattern_list = pattern_name
        else:
            raise ValueError(f"pattern_name should be str/list or None(using all available pattern)")

        if isinstance(hameln_graph_or_model, HamelnModel):
            graph = hameln_graph_or_model.hameln_graph
        elif isinstance(hameln_graph_or_model, HamelnGraph):
            graph = hameln_graph_or_model
        else:
            raise ValueError(f"input should be HamelnModel/HamelnGraph")
        for name in pattern_list:
            pattern:HamelnPattern = self.all_pattern[name]
            pattern.rewrite_graph(graph)
        
        


HPM = HamelnPatternManager()



conv_conv_concat_bn_leakyrelu = HamelnPattern().construct_pattern_from_definition(
    op_type_list=["Conv", "Conv", "Concat", "BatchNormalization", "LeakyRelu"],
    linkage=[[0, 2], [1, 2], [2, 3], [3, 4]])

def _conv_conv_concat_bn_leakyrelu_rewrite(hameln_graph: HamelnGraph, node_mapping):
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
        conv_0_bias = conv_0_bias[2]
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
        conv_1_bias = conv_1_bias[2]
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


conv_conv_concat_bn_leakyrelu.register_rewriter(_conv_conv_concat_bn_leakyrelu_rewrite)

HPM.register_pattern(
    pattern_name="conv_conv_concat_bn_leakyrelu",
    pattern=conv_conv_concat_bn_leakyrelu
)

m = HamelnModel("yolov5l_v3.onnx")




# %%

if __name__ == "__main__":

    m = HamelnModel("yolov5l_v3.onnx")

    HPM.rewrite(m)
    
    m.set_batch_size(32).set_nhwc_input_format().export("rewrite.onnx")
    
#%%
    
# # %%
# start_nodes_names = ["Conv_72", "Conv_71"]
# end_nodes_names = ["LeakyRelu_75"]

# # # start_nodes_names = ["Conv_47", "HardSigmoid_48"]

# # start_nodes_names = ["Mul_46"]
# # end_nodes_names = ["Conv_71", "Conv_72"]
# # end_nodes_names = ["Conv_71", "BatchNormalization_74"]

# # start_nodes_names = ["Conv_72", "Add_63"]
# # end_nodes_names = ["LeakyRelu_75"]

# all_nodes, start_nodes, end_nodes, inputs, outputs = g.extract_subgraph(start_nodes_names, end_nodes_names)

# for i in all_nodes:
#     print(f"current node: {i.get_name()}, from node: {[j.get_name() for j in i.from_node]}, to node: {[j.get_name() for j in i.to_node]}")
# print()

# # %%
# p = HamelnPattern().construct_pattern_from_subgraph(*g.extract_subgraph(start_nodes_names, end_nodes_names))

# # print(p.pattern_tree)

# # for i in p.pattern_tree.pattern_nodes:
# #     print(i)
# # print()

# p.pattern_tree.show()

# p = HamelnPattern().construct_pattern_from_definition(
#     op_type_list=["Conv", "Conv", "Concat", "BatchNormalization", "LeakyRelu"],
#     linkage=[[0, 2], [1, 2], [2, 3], [3, 4]])

# p.pattern_tree.show()
# #%%

# res = p.match(g)

# # %%
# res


# def _rewrite(hameln_graph: HamelnGraph, node_mapping):
#     conv_0: HamelnNode = hameln_graph.node[node_mapping[0]]
#     conv_1: HamelnNode = hameln_graph.node[node_mapping[1]]
#     concat_2: HamelnNode = hameln_graph.node[node_mapping[2]]
#     batchnormalization_3: HamelnNode = hameln_graph.node[node_mapping[3]]
#     leaky_relu_4: HamelnNode = hameln_graph.node[node_mapping[4]]
#     """
#         conv:
#         y1 = W @ input + B ( @ stands for conv )

#         bn: 
#         y2 = ((y1 - mean) / sqrt(var)) * weight + bias

#         ==》

#         fused_conv_bn
#         y2 = weight/sqrt(var) * y1 + bias -  mean * weight/sqrt(var)
#         = weight/sqrt(var) * W  @ input + (B - mean) * weight / sqrt(var) + bias


#         let scale = weight/sqrt(var)
#         then,
#             new_W = scale * W
#             new_B = scale * (B - mean) + bias
        
#     """

#     conv_0_input = conv_0.input
#     conv_0_weight = conv_0_input[1]
#     conv_0_weight_data = conv_0_weight.get_data()
#     if len(conv_0_input) == 3:
#         conv_0_bias = conv_0_bias[2]
#     else:
#         conv_0_bias = onnx_helper.make_tensor(
#             name=conv_0_weight.get_name().replace("weight", "bias") +
#             "_hameln",
#             data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
#                 np.float32)],
#             dims=[conv_0_weight_data.shape[0]],
#             vals=np.zeros((conv_0_weight_data.shape[0]), dtype=np.float32))
#         conv_0_bias = HamelnTensor(conv_0_bias)
#         conv_0.add_input(conv_0_bias)
#         hameln_graph.weight.append(conv_0_bias)
#     conv_0_bias_data = conv_0_bias.get_data()

#     conv_1_input = conv_1.input
#     conv_1_weight = conv_1_input[1]
#     conv_1_weight_data = conv_1_weight.get_data()
#     if len(conv_1_input) == 3:
#         conv_1_bias = conv_1_bias[2]
#     else:
#         conv_1_bias = onnx_helper.make_tensor(
#             name=conv_1_weight.get_name().replace("weight", "bias") +
#             "_hameln",
#             data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(
#                 np.float32)],
#             dims=[conv_1_weight_data.shape[0]],
#             vals=np.zeros((conv_1_weight_data.shape[0]), dtype=np.float32))
#         conv_1_bias = HamelnTensor(conv_1_bias)
#         conv_1.add_input(conv_1_bias)
#         hameln_graph.weight.append(conv_1_bias)
#     conv_1_bias_data = conv_1_bias.get_data()

#     bn_3_input = batchnormalization_3.input
#     bn_3_weight_data = bn_3_input[1].get_data()
#     bn_3_bias_data = bn_3_input[2].get_data()
#     bn_3_mean_data = bn_3_input[3].get_data()
#     bn_3_var_data = bn_3_input[4].get_data()

#     bn_weight_data1, bn_weight_data2 = np.split(bn_3_weight_data, 2)
#     bn_bias_data1, bn_bias_data2 = np.split(bn_3_bias_data, 2)
#     bn_mean_data1, bn_mean_data2 = np.split(bn_3_mean_data, 2)
#     bn_var_data1, bn_var_data2 = np.split(bn_3_var_data, 2)

#     scale1 = bn_weight_data1 / np.sqrt(bn_var_data1)
#     conv_0_weight_data = conv_0_weight_data * scale1.reshape(-1, 1, 1, 1)
#     conv_0_bias_data = (conv_0_bias_data -
#                         bn_mean_data1) * scale1 + bn_bias_data1

#     scale2 = bn_weight_data2 / np.sqrt(bn_var_data2)
#     conv_1_weight_data = conv_1_weight_data * scale2.reshape(-1, 1, 1, 1)
#     conv_1_bias_data = (conv_1_bias_data -
#                         bn_mean_data2) * scale2 + bn_bias_data2

#     conv_0_weight.set_data(conv_0_weight_data)
#     conv_0_bias.set_data(conv_0_bias_data)
#     conv_1_weight.set_data(conv_1_weight_data)
#     conv_1_bias.set_data(conv_1_bias_data)

#     left_leaky_relu_output_name = conv_0.output[0].get_name(
#     ) + "_leaky_relu_output"

#     left_leaky_relu = HamelnNode(
#         onnx_helper.make_node(op_type=leaky_relu_4.get_op_type(),
#                               inputs=[conv_0.output[0].get_name()],
#                               outputs=[left_leaky_relu_output_name],
#                               name=conv_0.get_name() + "_leaky_relu",
#                               alpha=leaky_relu_4.get_attribute()[0].f))

#     right_leaky_relu_output_name = conv_1.output[0].get_name(
#     ) + "leaky_relu_output"

#     right_leaky_relu = HamelnNode(
#         onnx_helper.make_node(op_type=leaky_relu_4.get_op_type(),
#                               inputs=[conv_1.output[0].get_name()],
#                               outputs=[right_leaky_relu_output_name],
#                               name=conv_1.get_name() + "_leaky_relu",
#                               alpha=leaky_relu_4.get_attribute()[0].f))

#     concat_output_name = leaky_relu_4.output[0].get_name()

#     last_concat = HamelnNode(
#         onnx_helper.make_node(
#             op_type=concat_2.get_op_type(),
#             inputs=[left_leaky_relu_output_name, right_leaky_relu_output_name],
#             outputs=[concat_output_name],
#             name=concat_2.get_name(),
#             axis=1))

#     # reconnect subgraph
#     HamelnNode.connect(conv_0, left_leaky_relu, clear_before_connect=True)
#     HamelnNode.connect(conv_1, right_leaky_relu, clear_before_connect=True)
#     HamelnNode.connect(left_leaky_relu, last_concat)
#     HamelnNode.connect(right_leaky_relu, last_concat)
#     HamelnNode.connect(last_concat,
#                        leaky_relu_4.to_node[0],
#                        clear_before_connect=True)

#     remove_node = [concat_2, batchnormalization_3, leaky_relu_4]
#     insert_node = [left_leaky_relu, right_leaky_relu, last_concat]

#     return True, remove_node, insert_node


# p.register_rewriter(_rewrite)

## %%

# r = g.set_nhwc_input_format()

# p.rewrite_graph(g)
## %%

# m.set_batch_size(30)

# #%%
# m.export("tmp2.onnx")
# #%%

# onnx.save(m.model, "tmp.onnx")

#%%
g = m.hameln_graph
g.add_internal_tensor_to_graph_output()
m.export("tmp9.onnx")
# %%
# p.match(g)
# %%
for i in g.node[39:45]:
    print(i.get_name())
# %%

m.set_nhwc_input_format()
# %%
m.set_batch_size(18).set_nhwc_input_format()
# %%
m.export("tmp3.onnx")
# %%

m = HamelnModel("tmp3.onnx")
# %%

p.rewrite_graph(m.hameln_graph)
# %%

m.export("tmp6.onnx")
# %%
p.rewrite_model(m)

m.set_nhwc_input_format().set_batch_size(13).export("tmp5.onnx")
# %%
