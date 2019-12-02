# Code documentation

This document will briefly explain the purposes of the top clases in case you want to reuse this code in your project. This is not a library, so I don't consider that auto generating documentation is worth it. For more information on the class behaviour, comments on the code have been made. Also, you can open an issue for clarification on this repo.

## Game
This is the base class to represent a game. It is the most simple; containes a python-chess board which holds the moves made, the player color (POV of the game) and the date when the game was played. Also, this class implements the logic to get the state of the game (`get_result()` or `get_legal_moves()`). **After each call to the `move()` method, the turn will be switched**. This is intended as a way to allow users to decouple the agent play logic from this class (though you can extend this class, being `GameAgent` or `GameStockfish` examples of this).


A simple way to use would be:

```python3
from game import Game
g = Game(player_color=Game.WHITE)
g.move('e2e4')  # Will return True if the movement was made
True
# Note that the turn has been changed, so we cannot move whites
g.move('b2b4')
False
g.move('d7d6')
True
print(g.get_result())  # Will return 1 if whites win, -1 if not, 0 on draw.
None
```
### Extensions of Game (GameAgent and GameStockfish)
This classes are childs of Game. You can use them to make a game that has an AI oponent. `GameAgent` will use an `Agent` instance to make the moves, whereas `GameStockfish` will use a `Stockfish` one. This games will automatically move after each call to the `move` method.

```python3
from game import Game
from gamestockfish import GameStockfish
from stockfish import Stockfish

player_color = Game.BLACK  # Our color

stockfish_instance = Stockfish(color=Game.WHITE, 'path/to/stockfish/binary')
g = GameStockfish(stockfish_instance, player_color=player_color) # Our player color
g.move(Game.NULL_MOVE)  # Or whatever, this is to force Stockfish to open the game.
# Our turn. Stockfish will have moved after this.
g.move('e7e6')
```

`GameAgent` is similar to stockfish with the difference that we have to pass an `Agent` instance.


## Player

This is the base class to represent a player. This may not be used, it's purpose it's only to serve as an interface.

### Agent

This class represents an AI which uses a neural network as the backend. On creation, a Tensorflow graph will be created and initialized with the trained weights of a model (if any). Because of this (if you don't have much GPU memory) if you plan to have several instances playing at once I would recommend using the `AgentDistributed` class or firing up each instance on a separate process. This class is practical as is easy to instantiate for a single-process workflow (for example, playing agaisnt one on a web client).

The most important method for the user of this is `best_move()` which will return the best next possible move (UCI encoded).

### AgentDistributed

This class is meant for performance/efficiency, and it's mainly used during self-play training. It relies on a client/server architecture. Instead of having one created model for each object, requests through a socket are made to a `PredictWorker` object (will be further explained). This way, we only execute one model and parallelization is possible (useful for the tree search).

### Stockfish

The same as Agent but using a Stockfish instance. 


## Tree

This class is the base for representing a tree which uses Monte Carlo Tree Search (this is only an interface). You must extend this class if you want to implement your own MCTS.

### SelfPlayTree

This class implements a parallelized version of a MCTS. It is meant to serve `AgentDistributed` during self-play training phase to use itself to explore all possible futures. Internally, for each expanded node of the tree, it uses copies of the `AgentDistributed` instance to make the moves of the opponent which only uses the policy vector returned by the neural net.

This implementation isn't intended to interact with an environment (like `GameAgent` or `GameStockfish`).

For the search it uses N threads to explore the tree during M iterations, so with a good CPU, you should be able to use a high M number.

During the tests, each search (cycle of a thread) takes about 0.4 secs (on an Intel i5 7600K).

Here is an example on how to use the class.

```python3
from game import Game
from agentdistributed import AgentDistributed
import mctree

# Skipping arguments
agent = AgentDistributed(...)
game = Game(...)

connection_pool = agent.pool  # We can use the agent's one or instantiate another.
tree = mctree.SelfPlayTree(game, connection_pool, threads=24)
tree.search_move(agent, max_iters=1600)  # Will return 'e2e4' (for example)
```

## PredictWorker

This fires up two threads. One will be listening to all new connections from the clients (`AgentDistributed`) and storing them in a list whileas and the other will be continually taking all data recieved, making predictions using the neural network and sending them back to the clients.

By default it will listen on `localhost:9999`, but you can use the address you want.

```python3
from predict_worker import PredictWorker

ENDPOINT = ('localhost', 2222)
worker = PredictWorker(model_path='path/to/model/model.h5', endpoint=ENDPOINT)
worker.start()

...  # Make other things

worker.stop()  # Stops the threads and closes all the connections. After this the clients must renew their connections.

...  # Make other things, training the model, for example

worker.reload_model('path/to/model/model.h5')

worker.start()   # Serve again

...  # Make other things

worker.stop()   # Kill the worker before exit.
```




