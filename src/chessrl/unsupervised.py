from agent import Agent
from gamestockfish import GameStockfish
from gameagent import GameAgent
from dataset import DatasetGame
from timeit import default_timer as timer
from lib.logger import Logger

import random
import argparse
import os
import traceback
import psutil
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def process_initializer():
    """ Initializer of the training threads in in order to detect if there
    is a GPU available and use it. This is needed to initialize TF inside the
    child process memory space."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_model_path(directory):
    """ Finds all the .h5 files (neural net weights) and returns the path to
    the most trained version. If there are no models, returns a default model
    name (model-0.h5)

    Parameters:
        directory: str. Directory path in which the .h5 files are contained

    Returns:
        path: str. Path to the file (directory+'/model-newest.h5')
    """

    path = directory + "/model-0.h5"

    # Model name
    models = [f for f in os.listdir(directory) if f.endswith("h5")]

    if len(models) > 0:
        # get greater version
        max_v = max([m.split("-")[1] for m in models])
        m = [model for model in models if model.endswith(max_v)][0]
        path = directory + "/" + m

    return path


def play_game_job(id: int, model_path, return_dict, mcts_iters,
                  stockfish_depth):
    """ Plays a game and returns the result..

    Parameters:
        id: Play ID (i.e. worker ID).
        model_path: path to the .h5 model. If it not exists, it will play with
        a fresh one.
        return_dict: dict. Shared variable in which each worker stores its
        result.
    """
    process_initializer()

    logger = Logger.get_instance()

    agent_is_white = True if random.random() <= .5 else False
    chess_agent = Agent(color=agent_is_white)
    # game_env = GameStockfish(player_color=agent_is_white,
    #                          stockfish='../../res/stockfish-10-64',
    #                          stockfish_depth=stockfish_depth)

    try:
        game_env = GameAgent(model_path, agent_is_white)
    except OSError:
        logger.error("Game agent not found. Exiting...")
        return

    try:
        chess_agent.load(model_path)
    except OSError:
        logger.warning("Model for play not found, using a random one.")

    logger.info(f"Starting game {id}")

    try:
        timer_start = timer()
        while game_env.get_result() is None:
            agent_move = chess_agent.best_move(game_env, real_game=False,
                                               max_iters=mcts_iters)
            game_env.move(agent_move)
        timer_end = timer()

        logger.info(f"Game {id} done. Result: {game_env.get_result()}. "
                    f"took {round(timer_end-timer_start, 2)} secs")
    except Exception:
        logger.error(traceback.format_exc())

    d = DatasetGame()
    d.append(game_env)
    game_env.tearup()

    return_dict[id] = str(d)


def train_job(model_dir, dataset_string):
    """ Loads (or creates, if not found) a model from model_dir, trains it
    and saves the results. This function has to be used inside a child process.

    Parameters:
        model_dir: str. Directory which contains the model
        dataset_string: str. DatasetGame serialized as a string.
    """
    logger = Logger.get_instance()

    process_initializer()

    data_train = DatasetGame()
    data_train.loads(dataset_string)

    model_path = get_model_path(model_dir)

    logger.info("Loading the agent...")
    chess_agent = Agent(color=True)
    try:
        chess_agent.load(model_path)
    except OSError:
        logger.warning("Model not found, training a fresh one.")
    chess_agent.train(data_train, logdir=model_dir, epochs=1)
    logger.info("Saving the agent...")
    chess_agent.save(model_path)


def train(model_dir, workers=1, save_plays=True,
          mcts_iters=900, stockfish_depth=15):
    """ Plays N concurrent games, save the results and then trains and saves
    the model.

    Parameters:
        model_dir: str. Directory where the neural net weights and training
            logs will be saved.
        games: number of games that will be played before training the model.
        workers: number of concurrent games (workers which will play the games)
        save_plays: Whether to save the results of the games before training
    """
    logger = Logger.get_instance()

    # If the dir does not exist, create it.
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_path = get_model_path(model_dir)

    logger.info(f"Setting up {workers} concurrent games.")

    dataset_games = DatasetGame()

    procs = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    for i in range(workers):
        proci = multiprocessing.Process(target=play_game_job,
                                        args=(i,
                                              model_path,
                                              return_dict,
                                              mcts_iters,
                                              stockfish_depth)
                                        )
        procs.append(proci)
        proci.start()

    for p in procs:
        p.join()

    for r in return_dict.values():
        di = DatasetGame()
        di.loads(r)
        dataset_games.append(di)

    logger.debug("GAMES PLAYED: {}".format(str(dataset_games)))

    if save_plays:
        logger.info("Storing the train recorded games")
        dataset_games.save(model_dir + '/gameplays.json')

    p = multiprocessing.Process(target=train_job,
                                args=(model_dir, str(dataset_games),))
    p.start()
    p.join()


def main():
    parser = argparse.ArgumentParser(description="Plays some chess games,"
                                     "stores the result and trains a model.")
    parser.add_argument('model_dir', metavar='modeldir',
                        help="where to store (and load from)"
                        "the trained model and the logs")
    parser.add_argument('--workers', metavar='workers', type=int,
                        default=1,
                        help="Number of processes to play the games "
                        "(the same as concurrent games)"
                        " (default = 1)")
    parser.add_argument('--train_rounds', metavar='train_rounds', type=int,
                        default=1,
                        help="Number of training cycles")
    parser.add_argument('--no_save_plays',
                        action='store_false',
                        default=True,
                        help="Whether you want to record the training plays."
                        "Default: true")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Log debug messages on screen. Default false.")
    parser.add_argument('--stockfish_depth', metavar='stockfish', type=int,
                        default=15,
                        help="Stockfish tree search depth. (Default = 15)")
    parser.add_argument('--mcts_iters', metavar='mcts', type=int,
                        default=900,
                        help="Max number of iterations of the MTCS during"
                        "playing phase.")

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    multiprocessing.set_start_method('spawn', force=True)

    # Recording and saving dataset
    logger.info("Starting training program.")
    for i in range(args.train_rounds):
        logger.info(f"Starting round {i+1} of {args.train_rounds}")
        mem_before = psutil.Process().memory_percent()

        train(args.model_dir,
              workers=args.workers,
              save_plays=args.no_save_plays,
              mcts_iters=args.mcts_iters,
              stockfish_depth=args.stockfish_depth
              )

        mem_after = psutil.Process().memory_percent()
        logger.debug(f"RAM INCREMENT {mem_after - mem_before}")


if __name__ == "__main__":
    main()
