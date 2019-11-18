import numpy as np

from game import Game
from player import Player
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor


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

    def __init__(self, state: 'Game', parent=None):
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
            value = (self.value / (1 + self.visits)) +\
                C * self.prior *\
                    (np.sqrt(self.parent.visits) / (1 + self.visits))
        return value

    def get_best_child(self):
        best = np.argmax([c.get_value() for c in self.children])
        return self.children[best]

    def __del__(self):
        del(self.state)
        self.parent = None
        self.children = None


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
            self.root = Node(root.get_copy())

        self.root.visits = 1

    def select(self, agent: Player, max_iters=200, verbose=False, noise=True):
        """ Explores and selects the best next state to choose from the root
        state

        Parameters:
            agent: Player. Agent which will be used in the simulations agaisnt
            stockfish (the neural network).
            max_iters: int. Number of interations to run the algorithm.
            verbose: bool. Whether to print the search status.
            noise: bool, Whether to apply dirichlet noise to the policy.
        """
        if verbose:
            pbar = tqdm(total=max_iters)
        #for _ in range(max_iters):
        #    if verbose:
        #        pbar.update(1)
        #    current_node = self.root
        #    current_node = self.forward(current_node, agent)
        #    v = self.simulate(current_node, agent)
        #    self.backprop(current_node, v)
        with ThreadPoolExecutor(max_workers=4) as executor:
            for _ in range(max_iters):
                executor.append(self.explore_tree, node=self.root, agent=agent)

        if verbose:
            del(pbar)

        max_val = np.argmax(self.policy(self.root, noise=noise))

        # After the last play the oponent has played, so we use the before last
        # one
        b_mov = Game.NULL_MOVE
        try:
            b_mov = self.root.children[max_val].state.board.move_stack[-2]
        except IndexError:
            # Should get here if it is not the agent's turn. For example, at
            # the first turn if the agent plays blacks.
            pass
        return b_mov

    def explore_tree(self, node, agent):
        current_node = self.root
        current_node = self.forward(current_node, agent)
        v = self.simulate(current_node, agent)
        self.backprop(current_node, v)

    def forward(self, node, agent):
        current_node = node
        while current_node.state.get_result() is None:
            if current_node.is_leaf:
                self.expand(current_node, agent=agent)
            else:
                current_node = current_node.get_best_child()
        return current_node

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

    def compute_policy(self, node: Node, noise=True):
        """ Calculates the policy vector of a state """
        # Select tau = 1 -> 0 (if number of moves > 30)
        nb_moves = len(node.state.board.move_stack)
        tau = 1
        if nb_moves >= 30:
            tau = nb_moves / (1 + np.power(nb_moves, 1.3))

        # Select argmax π(a|node) proportional to the visit count
        policy = np.array([np.power(v.visits, 1 / tau) for v in node.children]) / node.visits  # noqa:E501

        # apply random noise for ensuring exploration
        if noise:
            epsilon = 0.25
            policy = (1 - epsilon) * policy +\
                np.random.dirichlet([0.03] * len(node.children))
        return policy

    def _reset(self, node: Node):
        """ Sets all node references to None in order to GC can collect them
        well."""
        for c in node.children:
            self._reset(c)
        del(node)

    def __del__(self):
        self._reset(self.root)
