from agentdistributed import AgentDistributed
from predict_worker import PredictWorker
from game import Game

from timeit import default_timer as timer
import mctree
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Manager, Queue, Array
from multiprocessing.managers import BaseManager
from multiprocessing.connection import Connection
import lib.treeviewer as tv
import time
import gc
import os
import sys
from timeit import default_timer as timer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


endp = ('localhost', 9999)
worker = PredictWorker()
worker.start()
#
player_color = True
a1 = AgentDistributed(player_color, endpoint=endp)
gam = Game(player_color=player_color)

for i in range(2):
    start = timer()
    tree = mctree.SelfPlayTree(gam, a1.pool_conns, threads=8)
    bm = tree.search_move(a1, max_iters=900, verbose=False, ai_move=True)
    gam.move(bm[0])
    gam.move(bm[1])
    end = timer()

    elapsed = round(end - start, 2)
    print(f"\tMade move: {bm}, took: {elapsed} secs")


worker.stop()


worker.start()
print("**"*20)
print("ROUND 2")
print("**"*20)
a1 = AgentDistributed(player_color, endpoint=endp)
for i in range(2):
    start = timer()
    tree = mctree.SelfPlayTree(gam, a1.pool_conns, threads=8)
    bm = tree.search_move(a1, max_iters=900, verbose=False, ai_move=True)
    gam.move(bm[0])
    gam.move(bm[1])
    end = timer()

    elapsed = round(end - start, 2)
    print(f"\tMade move: {bm}, took: {elapsed} secs")

#    del(tree)
#
#    del(a1)
#    del(gam)
#    worker.stop()
#    del(worker)
#
#    gc.collect()
#    input("Key to next iter")

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


#manager = Manager()
#pipes = manager.Queue()
#pipes_lock = manager.Lock()
#for i in range(NUM_THREADS):
#    pipes.put(worker.get_pipe())



#worker.stop()







