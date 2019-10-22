from game import Game


class Player(object):
    """ This class represents contains the necessary methods all chess
    player objects must implement.
    """
    def __init__(self, color: bool, game: Game):
        """ Init

        Params:
            color: bool, Whether the player is white
        """
        if type(self) is Player:
            raise Exception('Cannot create Player Abstract class.')
        self.color = color
        self.game = game

        # Register me on the game
        game.add_player(self)

    def notify_turn(self, new_turn: bool):
        """
        Notifies the player the new turn. It must make a move if
        new_turn == self.color
        """
        raise Exception('Abstract class.')

    def make_move(self):
        """Makes the agent to make a move in the game. The agent should also
        pass the turn when it finishes (aka. notify the other players).
        """
        raise Exception('Abstract class.')
