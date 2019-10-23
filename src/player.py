class Player(object):
    """ This class represents contains the necessary methods all chess
    player objects must implement.
    """
    def __init__(self):
        if type(self) is Player:
            raise Exception('Cannot create Player Abstract class.')

    def make_move(self, game:'Game'):
        """Makes the agent to make a move in the game. The agent should also
        pass the turn when it finishes (aka. notify the other players).
        """
        raise Exception('Abstract class.')

