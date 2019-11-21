from game import Game

import random
import numpy as np


class RandomSimulation(object):
    """ Simulation of an agent agaisnt Other oponent """

    def __init__(self, game: Game):
        """ Parameters:
            agent: Player. Our agent.
            agent_policy: SimulationPolicy. Move selection strategy for the
            agent (for example, random selection.)
            stockfish_bin_path: str, PATH to the stockfish engine.
        """
        self.game = game

    def run(self, max_moves=100, repetitions=1):
        """ Runs the game simulation for max_moves """
        results = []
        for i in range(repetitions):
            n_mov = 0
            # While the game is not finished and the nb of moves is low
            while n_mov < max_moves and self.game.get_result() is None:
                self.game.move(random.choice(self.game.get_legal_moves()))
                n_mov += 1

            result = self.game.get_result()
            if n_mov > max_moves:
                result = 0  # Draw
            results.append(result)

        return np.mean(results)
