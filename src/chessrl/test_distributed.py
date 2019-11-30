from agentdistributed import AgentDistributed
from predict_worker import PredictWorker
from game import Game

from timeit import default_timer as timer
import mctree
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Queue, Array
from multiprocessing.managers import BaseManager
from multiprocessing.connection import Connection
import lib.treeviewer as tv
import time





worker = PredictWorker()
worker.start()

player_color = False
a1 = AgentDistributed(player_color, worker)
gam = Game(player_color=player_color)

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


#manager = Manager()
#pipes = manager.Queue()
#pipes_lock = manager.Lock()
#for i in range(NUM_THREADS):
#    pipes.put(worker.get_pipe())


if player_color is False:
    gam.move(a1.best_move(gam, real_game=True))

print(f" AGENT_ MOVE: {a1.best_move(gam, verbose=True, ai_move=True, max_iters=100)}")

worker.stop()






#print(a1.predict_outcome(gam))

#pool = [worker.get_pipe() for _ in range(10)]
#tree = mctree.Tree(gam, pool)
##
#start = timer()
#bm = tree.search_move(a1, verbose=True, max_iters=900)
#end = timer()
#elap = round(end - start, 2)
##
#print(bm)
#print(f"TOOK {elap} seconds")
#tv.draw_tree_html(tree)


# while gam.get_result() is None:
#     start = timer()
#     bm = a1.best_move(gam, real_game=False, max_iters=100, verbose=True)
#     end = timer()
#     gam.move(bm)
#
#     elap = round(end - start, 2)
#     print(f"My move took: {elap} seconds")

print(f"{gam.get_history()}")

