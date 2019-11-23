from gamestockfish import GameStockfish
from stockfish import Stockfish
from dataset import DatasetGame
from lib.logger import Logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import argparse
import numpy as np


def play_game(stockfish_bin, dataset, depth=1, tqbar=None, random_dep=False):
    is_white = True if np.random.random() <= .5 else False
    game_depth = depth
    player_depth = depth

    if random_dep:
        game_depth = int(np.random.normal(depth, 1))
        player_depth = int(np.random.normal(depth, 1))

    g = GameStockfish(stockfish=stockfish_bin,
                      player_color=is_white, stockfish_depth=game_depth)
    stockf = Stockfish(is_white, stockfish_bin, player_depth)

    while g.get_result() is None:
        bm = stockf.best_move(g)
        g.move(bm)

    dataset.append(g)
    if tqbar is not None:
        tqbar.update(1)

    # Kill stockfish processes
    g.tearup()
    stockf.kill()


def gen_data(stockfish_bin, save_path, num_games=100, workers=2,
             random_dep=False):
    logger = Logger.get_instance()
    d = DatasetGame()
    pbar = tqdm(total=num_games)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(num_games):
            executor.submit(play_game,
                            stockfish_bin=stockfish_bin,
                            dataset=d,
                            tqbar=pbar)
    pbar.close()
    logger.info("Saving dataset...")
    d.save(save_path)


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
    parser.add_argument('--random_depth',
                        action='store_true',
                        default=False,
                        help="Use normal distribution of depths with "
                        "mean --depth.")
    parser.add_argument('--debug',
                        action='store_true',
                        default=False,
                        help="Log debug messages on screen. Default false.")

    args = parser.parse_args()

    logger = Logger.get_instance()
    logger.set_level(1)

    if args.debug:
        logger.set_level(0)

    gen_data(args.stockfish_bin, args.data_path, args.games)


if __name__ == "__main__":
    main()
