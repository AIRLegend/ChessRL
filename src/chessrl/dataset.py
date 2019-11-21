import game

import json


class DatasetGame(object):
    """
    This class holds several games and provides operations to
    serialize/deserialize them as a JSON file. Also, it takes a game and
    returns it as the expanded game.
    """
    def __init__(self):
        self.games = []

    def augment_game(self, game_base):
        """ Expands a game. For the N movements of a game, it creates
        N games with each state + the final result of the original game +
        the next movement (in each state).
        """
        hist = game_base.get_history()
        moves = hist['moves']
        result = hist['result']
        date = hist['date']
        p_color = hist['player_color']

        augmented = []

        g = game.Game(date=date, player_color=p_color)

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
            g = game.Game(date=item['date'])
            if len(item['moves']) > 0:
                for m in item['moves']:
                    g.move(m)
                self.games.append(g)

    def loads(self, string):
        gamess = json.loads(string)
        for item in gamess:
            g = game.Game(date=item['date'], player_color=item['player_color'])
            if len(item['moves']) > 0:
                for m in item['moves']:
                    g.move(m)
                self.games.append(g)

    def save(self, path):
        dataset_existent = DatasetGame()
        try:
            dataset_existent.load(path)
        except FileNotFoundError:
            pass

        union_games = dataset_existent.games + self.games

        games = [x.get_history() for x in union_games]
        with open(path, 'w') as f:
            json.dump(games, f)

    def append(self, other):
        """ Appends a game (or another Dataset) to this one"""
        if isinstance(other, game.Game):
            self.games.append(other)
        elif isinstance(other, DatasetGame):
            self.games.extend(other.games)

    def __str__(self):
        games = [x.get_history() for x in self.games]
        return json.dumps(games)

    def __add__(self, other):
        """ Appends a game (or another Dataset) to this one"""
        self.append(other)
        return self

    def __iad__(self, other):
        """ Appends a game (or another Dataset) to this one"""
        return self.__add__(other)

    def __len__(self):
        return len(self.games)
