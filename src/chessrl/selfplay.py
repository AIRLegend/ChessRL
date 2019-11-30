from agentdistributed import AgentDistributed
from agent import Agent
from game import Game
from predict_worker import PredictWorker
from lib.logger import Logger

from dataset import DatasetGame
from timeit import default_timer as timer
from tensorflow.keras import backend as K
import tensorflow as tf

import argparse
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

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


def play_game(worker, agent):
    logger = Logger.get_instance()

    player_color = True if random.random() >= 0.5 else False

    gam = Game(player_color=player_color)
    agent.color = player_color

    if player_color is False:
        # Make the oponent move
        gam.move(agent.best_move(gam, real_game=True))

    # Play until finish
    while gam.get_result() is None:
        start = timer()
        bm, am = agent.best_move(gam, real_game=False, ai_move=True,
                                 max_iters=900)
        gam.move(bm)  # Make our move
        gam.move(am)  # Make oponent move
        end = timer()
        elapsed = round(end - start, 2)
        logger.debug(f"\tMade move: {bm}, took: {elapsed} secs")
        break
    logger.debug(gam.get_history())
    return gam


def train_model(game, model_path, model_dir):
    K.clear_session()
    K.set_floatx('float16')

    chess_agent = Agent(True, model_path)

    data_train = DatasetGame()
    data_train.append(game)

    chess_agent.train(data_train, logdir=model_dir, epochs=1,
                      validation_split=0, batch_size=1)
    chess_agent.save(model_path)

    K.clear_session()


def main():
    parser = argparse.ArgumentParser(description="Plays some chess games,"
                                     "stores the result and trains a model.")
    parser.add_argument('model_dir', metavar='modeldir',
                        help="where to store (and load from)"
                        "the trained model and the logs")
    parser.add_argument('--games', metavar='games', type=int,
                        default=1)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Log debug messages on screen. Default false.")

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    model_path = get_model_path(args.model_dir)
    worker = PredictWorker(model_path)
    agent = AgentDistributed(True, worker)

    games = 1
    for i in range(games):
        worker.start()
        gam = play_game(worker, agent)
        worker.stop()
        worker.flush_pipes()

        logger.debug(f"Training {i}: ")
        train_model(gam, model_path, args.model_dir)

        logger.info(f"Game {i} done")
        worker.reload_model(model_path)


if __name__ == "__main__":
    main()
