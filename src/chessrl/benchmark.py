""" This script serves as a way to get tangible metrics about how well a
trained agent behaves agaisnt Stockfish. For that, several games are played
and a summary of them is returned.
"""

from agent import Agent
from gamestockfish import GameStockfish
from timeit import default_timer as timer
from lib.logger import Logger
from concurrent.futures import ProcessPoolExecutor

import random
import os
import traceback
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
    tf.keras.backend.clear_session()


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


def play_game_job(id: int, model_path, stockfish_depth, log=False):
    """ Plays a game and returns the result..

    Parameters:
        id: Play ID (i.e. worker ID).
        model_path: path to the .h5 model. If it not exists, it will play with
        a fresh one.
        stockfish_depth: int. Difficulty of stockfish.
    """
    logger = Logger.get_instance()

    agent_is_white = True if random.random() <= .5 else False
    chess_agent = Agent(color=agent_is_white)
    game_env = GameStockfish(player_color=agent_is_white,
                             stockfish='../../res/stockfish-10-64',
                             stockfish_depth=stockfish_depth)

    try:
        chess_agent.load(model_path)
    except OSError:
        logger.error("Model not found. Exiting.")
        return None

    if log:
        logger.info(f"Starting game {id}")

    try:
        timer_start = timer()
        while game_env.get_result() is None:
            agent_move = chess_agent.best_move(game_env, real_game=True)
            game_env.move(agent_move)
        timer_end = timer()

        if log:
            logger.info(f"Game {id} done. Result: {game_env.get_result()}. "
                        f"took {round(timer_end-timer_start, 2)} secs")
    except Exception:
        logger.error(traceback.format_exc())

    res = {'color': agent_is_white, 'result': game_env.get_result()}
    game_env.tearup()
    return res


def benchmark(model_dir, workers=1, games=10, stockfish_depth=10, log=False):
    """ Plays N games and gets stats about the results.

    Parameters:
        model_dir: str. Directory where the neural net weights and training
            logs will be saved.
        workers: number of concurrent games (workers which will play the games)
    """
    multiprocessing.set_start_method('spawn', force=True)

    if log:
        logger = Logger.get_instance()
        logger.info(f"Setting up {workers} concurrent games.")

    model_path = get_model_path(model_dir)

    with ProcessPoolExecutor(workers, initializer=process_initializer)\
            as executor:

        results = []
        for i in range(games):
            results.append(executor.submit(play_game_job, *[i,
                                                            model_path,
                                                            stockfish_depth]))
    results = [r.result() for r in results]
    if log:
        logger.debug("Calculating stats.")
    won = [1
           if x['color'] is True and x['result'] == 1
           or x['color'] is False and x['result'] == -1 else 0  # noqa:W503
           for x in results]

    if log:
        print("##################### SUMMARY ###################")
        print(f"Games played: {games}")
        print(f"Games won: {len([x for x in won if x == 1])}")
        print(f"Games drawn: {len([x for x in results if x['result'] == 0])}")
        print("#################################################")

    return dict(played=games, won=len([x for x in won if x == 1]),
                drawn=len([x for x in results if x['result'] == 0]))


if __name__ == "__main__":
    benchmark('../../data/models/model1-unsuperv', workers=2, log=True)
