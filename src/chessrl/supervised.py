from agent import Agent
from dataset import DatasetGame
from lib.logger import Logger

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


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


def train(model_dir, dataset_path, epochs=1, batch_size=8):
    """ Loads (or creates, if not found) a model from model_dir, trains it
    and saves the results.

    Parameters:
        model_dir: str. Directory which contains the model
        dataset_string: str. DatasetGame serialized as a string.
    """
    logger = Logger.get_instance()

    logger.info("Loading dataset")

    data_train = DatasetGame()
    data_train.load(dataset_path)

    model_path = get_model_path(model_dir)

    logger.info("Loading the agent...")
    chess_agent = Agent(color=True)
    try:
        chess_agent.load(model_path)
    except OSError:
        logger.warning("Model not found, training a fresh one.")
    chess_agent.train(data_train, logdir=model_dir, epochs=epochs,
                      validation_split=0.25, batch_size=batch_size)
    logger.info("Saving the agent...")
    chess_agent.save(model_path)


def main():
    parser = argparse.ArgumentParser(description="Plays some chess games,"
                                     "stores the result and trains a model.")
    parser.add_argument('model_dir', metavar='modeldir',
                        help="where to store (and load from)"
                        "the trained model and the logs")
    parser.add_argument('data_path', metavar='datadir',
                        help="Path of .JSON dataset.")
    parser.add_argument('--epochs', metavar='epochs', type=int,
                        default=1)
    parser.add_argument('--bs', metavar='bs', help="Batch size. Default 8", type=int,
                        default=8)
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Log debug messages on screen. Default false.")

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    train(args.model_dir, args.data_path, args.epochs, args.bs)


if __name__ == "__main__":
    main()
