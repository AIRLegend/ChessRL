from agent import Agent
from game import GameStockfish
from dataset import DatasetGame
from timeit import default_timer as timer
import time
import random

import argparse
import concurrent.futures
import logging
import os
import traceback

import psutil

from tensorflow.keras import backend as K


def process_initializer():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0: 
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
    # Model name
    models = [f for f in os.listdir(directory) if f.endswith("h5")]

    path = directory + "/model-0.h5"

    if len(models) > 0:
        # get greater version
        max_v = max([m.split("-")[1] for m in models])
        m = [model for model in models if model.endswith(max_v)][0]
        path = directory + "/" + m

    return path


def play_game(datas: DatasetGame, id):
    """ Plays a game and store the results in datas.

    Parameters:
        datas: DatasetGame. Where the game history will be stored.
        id: Play ID.
    """
    logger = logging.getLogger("chessrl-train")

    agent_is_white = True if random.random() <= .5 else False

    chess_agent = Agent(color=agent_is_white)
    game_env = GameStockfish(player_color=agent_is_white,
                            stockfish='../../res/stockfish-10-64')

    logger.info(f"Starting game {id}")

    try:
        timer_start = timer()
        while game_env.get_result() is None:
            agent_move = chess_agent.best_move(game_env).uci()
            game_env.move(agent_move)
        timer_end = timer()

        logger.info(f"Game {id} done. Result: {game_env.get_result()}. "
                    f"took {round(timer_end-timer_start, 2)} secs")
    except Exception as e:
        logger.error(traceback.format_exc())

    datas += game_env
    game_env.tearup()


# @profile
def train(model_dir, games=1, threads=1, save_plays=True):
    """ Plays N concurrent games, save the results and then trains and saves
    the model.

    Parameters:
        model_dir: str. Directory where the neural net weights and training
            logs will be saved.
        games: number of games that will be played before training the model.
        threads: number of concurrent games (workers which will play the games)
    """

    datas = DatasetGame()
    logger = logging.getLogger("chessrl-train")

    logger.info(f"Set up {games} games distributed over {threads} threads.")

    with concurrent.futures.ProcessPoolExecutor(threads, 
            initializer=process_initializer) as executor:
        for i in range(games):
            executor.submit(play_game, *[datas, i])

    if save_plays:
        logger.info("Storing the train recorded games")
        datas.save(model_dir + '/gameplays.json')

    logger.info("Loading the agent...")

    return

    model_path = get_model_path(model_dir)
    chess_agent = Agent(color=True)
    try:
        chess_agent.load(model_path)
    except OSError:
        logger.warning("Model not found, starting a fresh one.")

    # Train the model
    logger.info("Training the agent...")
    chess_agent.train(datas, logdir=model_dir)

    # save model
    logger.info("Saving the agent...")
    chess_agent.save(model_path)

    #Free GPU memory
    #K.clear_session()


def main():
    parser = argparse.ArgumentParser(description="Plays some chess games,"
                                     "stores the result and trains a model.")
    # parser.add_argument('dataset_path', metavar='datasetpath', default=None,
    #                     help="where to store the recorded games dataset.")
    parser.add_argument('model_dir', metavar='modeldir',
                        help="where to store (and load from)"
                        "the trained model and the logs")
    # parser.add_argument('--verbose', metavar='verbose', type=int, default=2,
    #           help="Verbosity level: [0=nothing, 1=minimum, 2=all]")
    parser.add_argument('--games', metavar='games', type=int, default=1,
                        help="Number of games to play (default 1)")
    parser.add_argument('--threads', metavar='threads', type=int,
                        default=1,
                        help="Number of threads to play the games"
                        " (default = 1)")
    parser.add_argument('--train_rounds', metavar='train_rounds', type=int,
                        default=1,
                        help="Number of training cycles")
    parser.add_argument('--save_plays',
                        action='store_false',
                        help="Whether you want to record the training plays.")
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger("chessrl-train")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - "
                                  "%(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Recording and saving dataset
    logger.info("Starting training program.")
    for i in range(args.train_rounds):
        logger.info(f"Starting round {i} of {args.train_rounds}")
        mem_before = psutil.Process().memory_percent()
        train(args.model_dir,
              games=args.games,
              threads=args.threads,
              save_plays=args.save_plays)
        mem_after = psutil.Process().memory_percent()
        logger.debug(f"RAM INCREMENT {mem_after - mem_before}")


if __name__ == "__main__":
    main()
