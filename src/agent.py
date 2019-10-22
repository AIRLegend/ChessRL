import mctree

from player import Player
from game import Game


class Agent(Player):

    def __init__(self, game):
        super().__init__()
        self.game = game
        self.tree = mctree.Tree()

