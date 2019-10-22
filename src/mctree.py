import numpy as np


class Node(object):

    def __init__(self, state, parent=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.value = -1

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is not None

    def get_value(self):
        children_values = [c.get_value() for c in self.children]
        return np.mean(children_values)

    def evaluate(self):
        # TODO: pass custom evaluator (agent)
        return 0


class Tree(object):

    def __init__(self, root):
        self.root = root
