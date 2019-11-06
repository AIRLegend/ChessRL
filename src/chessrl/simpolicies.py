""" Policies for the agents which makes them select a particular movement from
the legal ones in the game.

This policies are not used for the MCTS that needs the Agent class (the neural
network one).
"""

import random

class SimulationPolicy(object):
    def __init__(self):
        if type(self) is SimulationPolicy:
            raise Exception('Cannot create Abstract class.')

    def best_movement(self, agent, game):
        pass


class NullPolicy(SimulationPolicy):
    """ Returns the best movement the agent thinks. """
    def __init__(self):
        super().__init__()

    def best_movement(self, agent, game):
        return self.oponent.best_move(self.game)


class RandomMovePolicy(SimulationPolicy):
    """ Returns a random movement from the possible ones."""
    def __init__(self):
        super().__init__()

    def best_movement(self, agent, game):
        best = None
        try:
            best = random.choice(game.get_legal_moves())
        except IndexError:
            pass
        return best
