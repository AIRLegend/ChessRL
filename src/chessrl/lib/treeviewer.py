from mctree import Tree, Node

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt

import tempfile
import webbrowser
import os

def draw_tree(tree: Tree):
    g = nx.DiGraph()

    labels_dict = {}
    __add_node(g, tree.root, labels_dict)

    pos = graphviz_layout(g, prog='dot')

    with plt.style.context("dark_background"):
        plt.figure(num="MCT-viz")
        plt.title("MCT visualization")
        nx.draw_networkx(g,
                         pos,
                         labels=labels_dict,
                         with_labels=True,
                         node_color='gray',
                         cmap='fire',
                         edge_color='gray',
                         font_color='w',
                         font_size=9,
                         node_size=60,
                         arrows=True)
        plt.show()


def __add_node(graph: nx.DiGraph, node: Node, labels_dict: dict):
    for c in node.children:
        graph.add_edge(hash(node), hash(c))
        # labels_dict[hash(c)] = {'val': c.value}
        labels_dict[hash(c)] = '' + "{0:.2g}".format(c.value)
        __add_node(graph, c, labels_dict)


def draw_tree_html(tree: Tree):
    """ Plots a tree on the browser """
    template_path = path = os.path.abspath(__file__ + "/../") + \
        "/static/template.html"

    tmp = tempfile.NamedTemporaryFile(delete=False)
    path = tmp.name + '.html'

    # load template
    with open(template_path, 'r') as file:
        filestr = file.read()

    tree_json = __json(tree.root)

    filestr = filestr.replace("{_DATA_}", str(tree_json))

    with open(path, 'w') as file:
        file.write(filestr)

    webbrowser.open('file://{}'.format(path))


def __json(node):
    node_json = {}
    node_json['text'] = {'name': f'V: {node.get_value()}'}
    node_json['children'] = [__json(n) for n in node.children]
    return node_json
