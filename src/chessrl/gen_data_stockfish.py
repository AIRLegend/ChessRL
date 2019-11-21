from gamestockfish import GameStockfish
from stockfish import Stockfish
from dataset import DatasetGame
from lib.logger import Logger

import argparse
import random


def play_game(stockfish_bin, dataset, depth=1):
    is_white = True if random.random() <= .5 else False

    g = GameStockfish(stockfish=stockfish_bin,
                      player_color=is_white, stockfish_depth=1)
    stockf = Stockfish(is_white, stockfish_bin, 1)

    while g.get_result() is None:
        bm = stockf.best_move(g)
        g.move(bm)

    dataset.append(g)

    # Kill stockfish processes
    g.tearup()
    stockf.kill()


def main():
    parser = argparse.ArgumentParser(description="Plays some chess games,"
                                     "stores the result and trains a model.")
    parser.add_argument('stockfish_bin', metavar='stockbin',
                        help="Stockfish binary path")
    parser.add_argument('data_path', metavar='datadir',
                        help="Path of .JSON dataset.")
    parser.add_argument('--games', metavar='games', type=int,
                        default=10)
    parser.add_argument('--depth', metavar='depth', type=int,
                        default=1, help="Stockfish tree depth.")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Log debug messages on screen. Default false.")

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    d = DatasetGame()

    for i in range(args.games):
        logger.info(f"Playing game {i} of {args.games}")
        play_game(args.stockfish_bin, d, depth=args.depth)

    logger.info("Saving dataset...")
    d.save(args.data_path)


if __name__ == "__main__":
    main()
