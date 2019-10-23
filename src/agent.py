import mctree

from player import Player


class Agent(Player):

    def __init__(self, color):
        super().__init__(color)
        #self.tree = mctree.Tree()

    def best_move(self, game:'Game') -> str:  # noqa: E0602, F821
        pass
