import numpy as np

from game import Game
from player import Player

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from timeit import default_timer as timer


VIRTUAL_LOSS = 1


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
        self.unexpanded_actions = state.get_legal_moves()
        self.parent = parent
        self.value = 0
        self.visits = 0
        self.prior = 1
        self.vloss = 0
        self.lock = Lock()

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_fully_expanded(self):
        return len(self.unexpanded_actions) == 0

    @property
    def is_terminal_state(self):
        return self.state.get_result() is not None

    @property
    def is_root(self):
        return self.parent is None

    def pop_unexpanded_action(self):
        return self.unexpanded_actions.pop()

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
            U = C*Prior*sqrt(sum(son.visitis))/1+self.visits
            Q = Expected value/visits ((mean value of its children))
        Being C a constant which makes the U (exploration part of the
        equation) more important.
        """
        C = 10
        value = 0
        if self.is_root:
            value = 99999999999  # Infinite to avoid division by 0
        else:
            value = (self.value / (1 + self.visits)) +\
                C * self.prior *\
                (np.sqrt(np.sum([c.visits for c in self.children])) / (1 + self.visits))  # noqa: E501
        return value - self.vloss

    def get_best_child(self):
        """Get the best child of this node.
        Returns:
            best: Node. Child with the max. PUCT value.
        """
        best = np.argmax([c.get_value() for c in self.children])
        return self.children[best]


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

    def search_move(self, agent, max_iters=200, verbose=False, noise=True,
                    ai_move=False):
        """ Explores and selects the best next state to choose from the root
        state

        Parameters:
            agent: Player. Agent which will be used in the simulations agaisnt
            stockfish (the neural network).
            max_iters: int. Number of interations to run the algorithm.
            verbose: bool. Whether to print the search status.
            noise: bool. Whether to add Dirichlet noise to the calc policy.
            ai_move: bool. Whether to return the move that AI will make after
            our best move
        """
        pass

    def explore_tree(self, node, agent, verbose=False):
        pass

    def select(self, node, agent):
        pass

    def expand(self, node, agent=None):
        pass

    def simulate(self, node: Node, agent: Player):
        pass

    def backprop(self, node: Node, value: float, remove_vloss=False):
        pass

    def compute_policy(self, node: Node, noise=True):
        pass


class SelfPlayTree(Tree):
    """ Monte Carlo Tree used for self playing.

    Parameters:
        root: Node or Game. Root state of the tree. You can pass a Node object
        with a Game as state or directly the game (it will make the Node).
    """
    def __init__(self, root, threads=6):
        super().__init__(root)
        self.num_threads = threads

    def search_move(self, agent, max_iters=200, verbose=False, noise=True,
                    ai_move=False):
        """ Explores and selects the best next state to choose from the root
        state

        Parameters:
            agent: Player. Agent which will be used in the simulations agaisnt
            stockfish (the neural network).
            max_iters: int. Number of interations to run the algorithm.
            verbose: bool. Whether to print the search status.
            noise: bool. Whether to add Dirichlet noise to the calc policy.
            ai_move: bool. Whether to return the move that AI will make after
            our best move
        """
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for _ in range(max_iters):
                executor.submit(self.explore_tree, node=self.root, agent=agent,
                                verbose=verbose)

        max_val = np.argmax(self.compute_policy(self.root, noise=noise))

        # After the last play the oponent has played, so we use the before last
        # one
        b_mov = Game.NULL_MOVE
        # The agent move which will be made after the last one of ours.
        agent_last_mov = Game.NULL_MOVE
        try:
            b_mov = self.root.children[max_val].state.board.move_stack[-2]
            agent_last_mov = self.root.children[max_val]\
                .state.board.move_stack[-1]
        except IndexError:
            # Should get here if it is not the agent's turn. For example, at
            # the first turn if the agent plays blacks.
            pass

        moves = str(b_mov), str(agent_last_mov)

        if not ai_move:
            moves = str(b_mov)
        return moves

    def explore_tree(self, node, agent, verbose=False):
        agent_copy = agent.get_copy()
        agent_copy.connect()

        start = timer()
        current_node = self.select(node, agent_copy)
        end = timer()
        v = self.simulate(current_node, agent_copy)
        self.backprop(current_node, v, remove_vloss=True)

        agent_copy.disconnect()

        elap = round(end - start, 2)
        if verbose:
            print(f"Elapsed on iteration: {elap} secs")

    def select(self, node, agent):
        current_node = node
        while not current_node.is_terminal_state:
            if not current_node.is_fully_expanded:
                current_node = self.expand(current_node, agent=agent)
                break
            else:
                current_node = current_node.get_best_child()

        # Wait if game is not updated yet
        with current_node.lock:
            current_node.vloss += VIRTUAL_LOSS

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
        new_state = node.state.get_copy()
        new_state.move(node.pop_unexpanded_action())
        # Move oponent
        if new_state.get_result() is None:
            bm = agent.best_move(new_state, real_game=True)
            new_state.move(bm)

        new_child = Node(new_state, parent=node)
        node.children.append(new_child)

        # If this node was the last one before fully expand the node
        # we calculate the priors of the children
        # (only do it once)
        if node.is_fully_expanded:
            self._update_prior(node, agent)

        return new_child

    def simulate(self, node: Node, agent: Player):
        """ Rollout from the current node until a final state.

        Parameters:
            node: Node. Game state from which the simulation will be run.
            agent: Agent. that will be used to play the games.
        Returns:
            results_sim: float, Result of the playout/predicted value from NN.
        """
        result = node.state.get_result()
        if result is None:
            result = agent.predict_outcome(node.state)

            # Random sim
            # sim = RandomSimulation(node.state.get_copy())
            # result = sim.run(repetitions=500)

        return result

    def backprop(self, node: Node, value: float, remove_vloss=False):
        """ Backpropagation phase of the algorithm.

        Parameters:
            node: Node. that will be added 1 to vi and the value obtained in
                the simulation process.
            value: float, value that will be added to all of the ancestors
                untill root.
            remove_vloss: Remove virtual loss from the parent nodes to allow
                the exploration of the same path by other threads
        """
        with node.lock:
            node.visits += 1
            node.value += value
            if remove_vloss:
                node.vloss -= VIRTUAL_LOSS

        if node.parent is not None:
            self.backprop(node.parent, value)

    def _update_prior(self, node, agent):
        """ Update the priors of the children nodes """
        priors = agent.predict_policy(node.state, mask_legal_moves=True)
        children = reversed(node.children)
        for p, n in zip(priors, children):
            n.prior = p

    def compute_policy(self, node: Node, noise=True):
        """ Calculates the policy vector given a game state """
        # Select tau = 1 -> 0 (if number of moves > 30)
        nb_moves = len(node.state.board.move_stack)
        tau = 1
        if nb_moves >= 30:
            tau = nb_moves / (1 + np.power(nb_moves, 1.3))

        # Select argmax π(a|node) proportional to the visit count
        policy = np.array([np.power(v.visits, 1 / tau) for v in node.children])\
            / np.power(node.visits, 1 / tau)

        # apply random noise for ensuring exploration
        if noise:
            epsilon = 0.25
            policy = (1 - epsilon) * policy +\
                np.random.dirichlet([0.03] * len(node.children))
        return policy
