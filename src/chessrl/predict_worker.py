from model import ChessModel
import netencoder

import numpy as np
import multiprocessing
from multiprocessing.connection import wait
import threading
import socket


class PredictWorker():
    """ This will run a separate process maintaining a model. Prediction
    requests will be made to this class.
    """

    def __init__(self,
                 model_path='../../data/models/model1-unsuperv/model-0.h5',
                 endpoint=('localhost', 9999)
                 ):
        self.model = ChessModel(weights=model_path)
        self.address = endpoint
        self.listener = None
        self.connections = []

    def start(self):
        """ Opens a socket to listen to new connections and creates two
        threads. One for listening to new clients and other for making the
        predictions.
        """
        if self.listener is None:
            self.listener = multiprocessing.connection.Listener(self.address)
            self.listener._listener._socket.settimeout(4)

            self.do_run = True
            self.predict_worker = threading.Thread(target=self.__work,
                                                   name="prediction_worker")
            self.predict_worker.start()
            self.conn_worker = threading.Thread(
                target=self.__accept_connections,
                name="connection_worker")
            self.conn_worker .start()

    def stop(self):
        """ Stops the threads, closes the socket and terminates all the
        open connections of the clients.
        """
        self.do_run = False
        for c in self.connections:
            c.close()
        self.connections = []
        self.listener.close()

        self.conn_worker.join()
        self.predict_worker.join()

        self.listener = None
        self.predict_worker = None
        self.conn_worker = None

    def reload_model(self, model_path):
        """ Loads a model to make predictions. This should be called with the
        worker stopped.

        Parameters:
            model_path: string. Route to the model weights (.h5 file).
        """
        self.model = ChessModel(weights=model_path)

    def __work(self):
        """ This method does the actual work of taking batches of requests and
        send the responses back to the clients.
        """
        while self.do_run:
            ready = wait(self.connections, timeout=0.0001)
            if not ready:
                continue
            data, result_conns = [], []
            try:
                for conn in ready:
                    while conn.poll():
                        g = conn.recv()
                        data.append(netencoder.get_game_state(g))
                        result_conns.append(conn)
            except (OSError, EOFError):
                # If the client closed the connection during the wait, we
                # discard it and continue.
                continue

            if len(data) > 0:
                data = np.asarray(data, dtype=np.float16)
                policies, values = self.model.predict(data)
                for conn, p, v in zip(result_conns, policies, values):
                    conn.send((p, float(v)))

    def __accept_connections(self):
        """ This method will accept all new connections and put them on
        the client list.
        """
        while self.do_run:
            try:
                new_con = self.listener.accept()
            except (socket.timeout, OSError):
                continue
            self.connections.append(new_con)
