from agent import Agent
from gameagent import GameAgent
from game import Game
from dataset import DatasetGame
from timeit import default_timer as timer

import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def play_game():
    player_color = True if random.random() > 0.5 else False
    player_color = False

    model_path = '../../data/models/model1-unsuperv/model-0.h5'

    agent = Agent(player_color, model_path)
    gam = GameAgent(model_path, player_color=player_color)

    if player_color is False:
        gam.move(Game.NULL_MOVE)  # Force agent to make the first move

    print(gam.get_history())
    # Play until finish
    while gam.get_result() is None:
        start = timer()
        bm = agent.best_move(gam, real_game=False, max_iters=100)
        gam.move(bm)
        end = timer()
        elapsed = round(end - start, 2)
        print(f"\tMade move: {bm}, took: {elapsed} secs")

    return gam


def train_model(game):
    model_path = '../../data/models/model1-unsuperv/model-0.h5'
    model_dir = '../../data/models/model1-unsuperv'
    chess_agent = Agent(color=True)
    chess_agent.load(model_path)

    data_train = DatasetGame()
    data_train.append(game)

    chess_agent.train(data_train, logdir=model_dir, epochs=1,
                      validation_split=0, batch_size=1)
    chess_agent.save(model_path)


def main():
    games = 20
    for i in range(games):
        gam = play_game()
        train_model(gam)
        print("Game {i} done")


if __name__ == "__main__":
    main()
