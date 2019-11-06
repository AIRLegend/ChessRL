import numpy as np
from game import Game
from player import Player
from tqdm import tqdm


class Node(object):
    """ Node from a Monte Carlo Tree.

    Attributes:
        state: Game. Game in a certain state.
        children: Array. Possible game states after applying the legal moves to
        the current state.
        parent: Node. Parent state of the current game.
        value: float. Expected reward of this node.
        visits: int. Number of times the node has been visited
        prior: float.
    """

    def __init__(self, state: Game, parent=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.value = 0
        self.visits = 0
        self.prior = 1

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    def get_ucb1(self):
        """ returns the UCB1 metric of the node. """
        C = 2
        value = 0
        if self.visits == 0:
            value = 99999999999  # Infinite to avoid division by 0
        else:
            # Vanilla MCTS
            # Return ucb1 score = vi + c * sqrt(log(N)/ni)
            value = self.value / self.visits + C * \
                np.sqrt(np.log(self.parent.visits) / self.visits)
        return value

    def get_value(self):
        """Returns the Q + U for the node using the prior probabilities
        given by the neural network.
            U \\propto P(s,a)/(1+visits);
            Q = Expected value/visits
        Being C a constant which makes the U (exploration part of the
        equation) more important.
        """
        C = 3
        value = 0
        if self.is_root:
            value = 99999999999  # Infinite to avoid division by 0
        else:
            value = (self.value / self.visits) +\
                C * self.prior *\
                    (np.sqrt(self.parent.visits) / (1 + self.visits))
        return value


class Tree(object):
    """ Monte Carlo Tree.

    Parameters:
        root: Node or Game. Root state of the tree. You can pass a Node object
        with a Game as state or directly the game (it will make the Node).
    """
    def __init__(self, root):
        if type(root) is Node:
            self.root = root
        else:
            self.root = Node(root)

        self.root.visits = 1

    def select(self, agent: Player, max_iters=200, verbose=False):
        """ Explores and selects the best next state to choose from the root
        state

        Parameters:
            agent: Player. Agent which will be used in the simulations agaisnt
            stockfish (the neural network).
            max_iters: int. Number of interations to run the algorithm.
            verbose: bool. Whether to print the search status.
        """
        current_node = self.root
        i = 0
        if verbose:
            print("Monte Carlo Tree Search running...")
            pbar = tqdm(total=max_iters)
        while(i < max_iters):
            i += 1
            if verbose:
                pbar.update(1)
            if current_node.is_leaf:
                if current_node.visits == 0:  # Is new
                    res = self.simulate(current_node, agent)
                    self.backprop(current_node, res)
                else:  # Is not new
                    self.expand(current_node, agent)
                    res = current_node.value
                    try:
                        max_child = np.argmax([x.value for x in
                                               current_node.children])
                        current_node = current_node.children[max_child]
                        res = self.simulate(current_node, agent)
                    except ValueError:  # If there were no children
                        pass
                    self.backprop(current_node, res)

                    # Return to root (finish cycle)
                    current_node = self.root
            else:
                ucbs = [u.get_value() for u in current_node.children]
                best_child = np.argmax(ucbs)
                current_node = current_node.children[best_child]

        if verbose:
            del(pbar)
        # Select the highest confidence one.
        max_val = np.argmax([v.get_value() for v in self.root.children])
        # After the last play the oponent has played, so we use the before last
        b_mov = Game.NULL_MOVE
        try:
            b_mov = self.root.children[max_val].state.board.move_stack[-2]
        except IndexError:
            # Should get here if it is not the agent's turn. For example, at
            # the first turn if the agent plays blacks.
            pass
        return b_mov

    def expand(self, node, agent=None):
        """
        From a given state (node), adds to itself all its children
        (game states after all the possible legal game moves are applied).
        Note that this process makes a move and assume that the game oponent
        moves after our move, resulting in the new state.

        Parameters:
            node: Node. Node which will be expanded.
        """
        legal_moves = [m for m in node.state.get_legal_moves()]
        new_states = []
        for m in legal_moves:
            new_state = node.state.get_copy()
            new_state.move(m)
            new_states.append(new_state)

        node.children = [Node(s, parent=node) for s in new_states]

        # If there is an agent, we calculate the prior probabilities
        # of selecting each possible move.
        if agent:
            # Returns policy distribution for LEGAL moves
            pi = agent.predict_policy(node.state)
            for i, c in enumerate(node.children):
                c.prior = pi[i]

    def simulate(self, node: Node, agent: Player):
        """ Rollout from the current node until a final state.

        Parameters:
            node: Node. Game state from which the simulation will be run.
            agent: Agent. that will be used to play the games (our Neural net).
        Returns:
            results_sim: float, Result of the simulations
        """
        # VANILLA
        # sim_pol = RandomMovePolicy()
        # sim = Simulation(agent, node.state.get_copy(), sim_pol)
        # results_sim = sim.run(max_moves=200)  # []

        # NEURAL NET
        return agent.predict_outcome(node.state)

    def backprop(self, node: Node, value: float):
        """ Backpropagation phase of the algorithm.

        Parameters:
            node: Node. that will be added 1 to vi and the value obtained in
            the simulation process.
            value: float, value that will be added to all of the ancestors
            untill root.
        """
        node.visits += 1
        node.value += value

        if node.parent is not None:
            self.backprop(node.parent, value)
