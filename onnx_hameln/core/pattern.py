import networkx as nx
from networkx.algorithms import isomorphism

from .hameln import HamelnModel, HamelnGraph


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
        assert self.pattern_graph is not None, "call compile graph before show"

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
        # TODO(chen.chen): I don't know here what exactly we need from subgraph...
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

        # TODO(chen.chen): wtf, I don't know why I add a status variable here, remove it
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
    all_pattern = {}

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kw)
        return cls._instance

    def __repr__(self):
        info = f"HamelnPatternManager with {len(HamelnPatternManager.all_pattern)} patterns:\n\n"
        if len(HamelnPatternManager.all_pattern) > 0:
            info += "\n".join([
                f"\t{i}" for i in list(HamelnPatternManager.all_pattern.keys())
            ])
        return info

    def get_available_pattern(self):
        return list(HamelnPatternManager.all_pattern.keys())

    def get_pattern(self, pattern_name):
        pattern = HamelnPatternManager.all_pattern.get(pattern_name, None)
        if pattern is None:
            raise ValueError(f"can not find pattern: {pattern_name}")
        return pattern

    def register_pattern(self, pattern_name, pattern):
        if pattern_name in HamelnPatternManager.all_pattern:
            raise KeyError(f"pattern: {pattern_name} exists")
        HamelnPatternManager.all_pattern[pattern_name] = pattern

    def rewrite(self, hameln_graph_or_model, pattern_name=None):
        if pattern_name is None:
            pattern_list = self.get_available_pattern()
        elif isinstance(pattern_name, str):
            pattern_list = [pattern_name]
        elif isinstance(pattern_name, list):
            pattern_list = pattern_name
        else:
            raise ValueError(
                "pattern_name should be str/list or None(using all available pattern)"
            )

        if isinstance(hameln_graph_or_model, HamelnModel):
            graph = hameln_graph_or_model.hameln_graph
        elif isinstance(hameln_graph_or_model, HamelnGraph):
            graph = hameln_graph_or_model
        else:
            raise ValueError("input should be HamelnModel/HamelnGraph")
        for name in pattern_list:
            pattern: HamelnPattern = HamelnPatternManager.all_pattern[name]
            pattern.rewrite_graph(graph)
