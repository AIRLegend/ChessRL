import threading


class AsyncWrapper(object):

    def __init__(self, game):
        self.game = game
        self.board = game.board
        self.done_moving_lock = threading.RLock()
        self.done_moving = False

    def move(self, movement):

        worker = threading.Thread(target=self.__async_task, args=(movement,))
        worker.start()

    def __async_task(self, movement):
        with self.done_moving_lock:
            self.game.move(movement)
            self.done_moving = True

    @property
    def turn(self):
        with self.done_moving_lock:
            turn = self.game.turn
        return turn

    def get_legal_moves(self):
        with self.done_moving_lock:
            legal = self.game.get_legal_moves()
        return legal

    def get_result(self):
        with self.done_moving_lock:
            result = self.game.get_result()
        return result

    def get_copy(self):
        with self.done_moving_lock:
            copy = self.game.get_copy()
        return copy
