from game import Game

import json


class DatasetGame(object):
    """
    Game records to train a neural
    """
    def __init__(self):
        self.games = []

    def get_data(self):
        """ For each game in the dataset, it expands it by cloning it and
        adding a movement (1 Game -> N games each one adding one more
        movement)
        """
        games_aug = []

        for g in self.games:
            games_aug.extend(self.augment_game(g))
        return games_aug

    def augment_game(self, game):
        hist = game.get_history()
        moves = hist['moves']
        result = hist['result']
        p_color = hist['player_color']

        if result is None:
            result = 0

        # start = -1 if p_color else 0

        augmented = []

        g = Game(player_color=p_color)

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
        if isinstance(other, Game):
            self.games.append(other)
        elif isinstance(other, DatasetGame):
            self.games.extend(other.games)

        return self

    def __iad__(self, other):
        return self.__add__(other)

    def __len__(self):
        """ Returns the length of the augmented dataset """
        return sum([len(g) for g in self.games])
