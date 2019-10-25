from mctree import Tree, Node

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt


def draw_tree(tree: Tree):
    g = nx.DiGraph()

    labels_dict = {}
    __add_node(g, tree.root, labels_dict)

    pos = graphviz_layout(g, prog='dot')

    plt.title("MCST visualization")
    plt.style.use('dark_background')
    nx.draw_networkx(g,
                     pos,
                     labels=labels_dict,
                     with_labels=True,
                     #node_color='skyblue',
                     cmap='fire',
                     edge_color='gray',
                     font_color='white',
                     font_size=9,
                     arrows=True)


def __add_node(graph: nx.DiGraph, node: Node, labels_dict: dict):
    for c in node.children:
        graph.add_edge(hash(node), hash(c))
        # labels_dict[hash(c)] = {'val': c.value}
        labels_dict[hash(c)] = '' + "{0:.2g}".format(c.value)
        __add_node(graph, c, labels_dict)
