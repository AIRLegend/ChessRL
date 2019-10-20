import game


class Node(object):

    def __init__(self, state):
        self.state = state
        self.children = []

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def evaluate(self):
        #TODO: pass custom evaluator (agent)
        return 0


class Tree(object):

    def __init__(self, root):
        self.root = root


