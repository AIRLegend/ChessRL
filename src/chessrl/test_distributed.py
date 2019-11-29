from agentdistributed import Agent
from predict_worker import PredictWorker
from gameagent import GameAgent
from game import Game

from timeit import default_timer as timer
import mctree
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Queue, Array
from multiprocessing.managers import BaseManager
from multiprocessing.connection import Connection
import lib.treeviewer as tv
import time


class PipePool:
    def __init__(self, pipes):
        self.manager = Manager()
        self.objs = self.manager.list(pipes)
        #self.free_idx = [i for i in range(len(pipes))]
        #self.extracted_idx = []
        #self.lock = Lock()
        self.free_indices = self.manager.Queue()
        self.extracted_indices = self.manager.Queue()
        for i in range(len(pipes)):
            self.free_indices.put(i)

    def pop(self):
        idx = self.free_indices.get()
        self.extracted_idx.put(idx)
        obj = self.objs[idx]
        return obj

    def put(self):
        idx = self.extracted_idx.pop()
        self.free_indices.put(idx)



worker = PredictWorker()
worker.start()

a1 = Agent(True, worker)
# a2 = Agent(False, worker)
# a2.pipe = pipes[0]

#gam = GameAgent(a2)
gam = Game()

# TEST THREADS PIPES
NUM_THREADS = 4
#manager = Manager()
#pipes = manager.list([worker.get_pipe() for i in range(NUM_THREADS)])


#def make_pred(game, agent, pipe):
#    print(pipe)
#    pipe.send(game)
#    print(pipe.recv())
#
#
#with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
#    for i in range(NUM_THREADS *2 ):
#        executor.submit(make_pred, game=gam, agent=a1, pipe=pipes[i])


manager = Manager()
pipes = manager.Queue()
pipes_lock = manager.Lock()
for i in range(NUM_THREADS):
    pipes.put(worker.get_pipe())



class SharedQueue:
    """ Proxies to shared objects"""
    def __init__(self, queue, lock):
        self.queue = queue
        self.lock = lock

    def get(self):
        with self.lock:
            succ = False
            while not succ:
                try:
                    obj = self.queue.get(timeout=0.1)
                    succ = True
                    print("PICKED")
                except Exception:
                    succ = False
        return obj

    def put(self, obj):
        with self.lock:
            succ = False
            while not succ:
                try:
                    self.queue.put(obj, timeout=0.1)
                    succ = True
                    #print("DROPPED")
                except Exception:
                    succ = False

# sq = SharedQueue(pipes, pipes_lock)
#
# def make_pred(game, sq):
#     pipe = sq.get()
#
#     print(pipe)
#     pipe.send(game)
#     print(pipe.recv())
#
#     sq.put(pipe)
#
#
# with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
#     for i in range(NUM_THREADS * 30):
#         executor.submit(make_pred, game=gam, sq=sq)
#











#print(a1.predict_outcome(gam))

pool = [worker.get_pipe() for _ in range(10)]
tree = mctree.Tree(gam, pool)

start = timer()
bm = tree.search_move(a1, verbose=True, max_iters=900)
end = timer()
elap = round(end - start, 2)

print(bm)
print(f"TOOK {elap} seconds")
tv.draw_tree_html(tree)


# while gam.get_result() is None:
#     start = timer()
#     bm = a1.best_move(gam, real_game=False, max_iters=100, verbose=True)
#     end = timer()
#     gam.move(bm)
#
#     elap = round(end - start, 2)
#     print(f"My move took: {elap} seconds")

print(f"{gam.get_history()}")

