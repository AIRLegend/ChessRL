from agent import Agent
from gameagent import GameAgent
from game import Game

def main():
    agent = Agent(True, '../../data/models/model1-superv/model-0.h5')

    gam = GameAgent(agent, player_color=True)

    while gam.get_result() is None:
        bm = agent.best_move(gam, real_game=False, max_iters=900)
        print(bm)
        gam.move(bm)
        print(gam.get_history())


if __name__ == "__main__":
    main()
