from gameagent import GameAgent  # noqa:E402


class GameWrapper(object):
    _wrapper = None

    def __init__(self, player_color=True,
                 agent_path='../../data/models/model0/model-0.h5'):
        self.game = GameAgent(agent_path,
                              player_color=player_color)

    def get_instance(player_color=True, agent_path=None):
        if GameWrapper._wrapper is None:
            if agent_path is not None:
                GameWrapper._wrapper = GameWrapper(player_color,
                                                   agent_path=agent_path)
            else:
                GameWrapper._wrapper = GameWrapper(player_color)

        return GameWrapper._wrapper

    def destroy_instance():
        GameWrapper._wrapper.game.tearup()
        GameWrapper._wrapper = None

    @property
    def fen(self):
        return self.game.get_fen()

    @property
    def player_color(self):
        return self.game.player_color

    def move(self, uci):
        self.game.move(uci)

    def get_result(self):
        return self.game.get_result()

    def get_history(self):
        return self.game.get_history()

    def get_result_str(self):
        res = self.game.get_result()
        ret = "In progress..."
        if res is not None:
            if res == 1:
                ret = "Whites win"
            elif res == -1:
                ret = "Blacks win"
            else:
                ret = "Draw"
        return ret
