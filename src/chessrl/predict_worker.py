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
        self.predict_worker = None
        self.do_run = True

    def start(self):
        if self.predict_worker is None:
            self.do_run = True
            self.predict_worker = threading.Thread(target=self.__work,
                                                name="prediction_worker")
            self.predict_worker.start()

    def stop(self):
        self.do_run = False
        self.predict_worker.join()
        self.predict_worker = None
        self.flush_pipes()

    def flush_pipes(self):
        for p in self.pipes:
            p.close()
        self.pipes.clear()

    def reload_model(self, model_path):
        self.model = ChessModel(weights=model_path)

    def get_pipe(self):
        mine, yours = multiprocessing.Pipe()
        self.pipes.append(mine)
        return yours

    def __work(self):
        while self.do_run:
            ready = wait(self.pipes, timeout=0.0001)
            if not ready:
                continue
            data, result_pipes = [], []
            for pipe in ready:
                while pipe.poll():
                    g = pipe.recv()
                    data.append(netencoder.get_game_state(g))
                    result_pipes.append(pipe)

            data = np.asarray(data, dtype=np.float16)
            policies, values = self.model.predict(data)
            for pipe, p, v in zip(result_pipes, policies, values):
                pipe.send((p, float(v)))
