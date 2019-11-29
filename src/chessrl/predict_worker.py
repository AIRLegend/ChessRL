from model import ChessModel
import netencoder

import numpy as np
import multiprocessing
from multiprocessing.connection import wait
import threading


class PredictWorker():
    """ This will run a separate process maintaining a model. Prediction
    requests will be made to this class.
    """

    def __init__(self,
                 model_path='../../data/models/model1-unsuperv/model-0.h5',
                 **kwargs):
        self.kwargs = kwargs
        self.model = ChessModel(weights=model_path)
        self.pipes = []

    def start(self):
        prediction_worker = threading.Thread(target=self.__work,
                                             name="prediction_worker")
        prediction_worker.start()

    def get_pipe(self):
        mine, yours = multiprocessing.Pipe()
        self.pipes.append(mine)
        return yours

    def __work(self):
        while True:
            ready = wait(self.pipes, timeout=0.0001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    g = pipe.recv()
                    data.append(netencoder.get_game_state(g))
                    # data.append(g)
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float16)
            policies, values = self.model.predict(data)
            for pipe, p, v in zip(result_pipes, policies, values):
                pipe.send((p, float(v)))
