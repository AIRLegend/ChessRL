from game import Game

import json


class DatasetGame(object):
    """
    This class holds several games and provides operations to
    serialize/deserialize them as a JSON file. Also, it takes a game and
    returns it as the expanded game.
    """
    def __init__(self):
        self.games = []

    def augment_game(self, game):
        """ Expands a game. For the N movements of a game, it creates
        N games with each state + the final result of the original game +
        the next movement (in each state).
        """
        hist = game.get_history()
        moves = hist['moves']
        result = hist['result']
        # p_color = hist['player_color']

        # if result is None:
        #     result = 0

        augmented = []

        g = Game()

        for m in moves:
            augmented.append({'game': g,
                              'next_move': m,
                              'result': result})
            g = g.get_copy()
            g.move(m)

        return augmented

    def load(self, path):
        games_file = []
        with open(path, 'r') as f:
            games_file = json.load(f)

        for item in games_file:
            g = Game()
            for m in item['moves']:
                g.move(m)
            self.games.append(g)

    def save(self, path):
        games = [x.get_history() for x in self.games]
        with open(path, 'w') as f:
            json.dump(games, f)

    def __add__(self, other):
        """ Appends a game (or another Dataset) to this one"""
        if isinstance(other, Game):
            self.games.append(other)
        elif isinstance(other, DatasetGame):
            self.games.extend(other.games)

        return self

    def __iad__(self, other):
        """ Appends a game (or another Dataset) to this one"""
        return self.__add__(other)

    def __len__(self):
        return len(self.games)
