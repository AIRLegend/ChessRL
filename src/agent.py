import Game
import mctree


class Agent(object):

    def __init__(self, game):
        self.game = game
        self.tree = mctree.Tree()


