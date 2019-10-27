import numpy as np
from game import Game
from simpolicies import RandomMovePolicy
from simulation import Simulation


class Node(object):

    def __init__(self, state: Game, parent=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.value = 0
        self.visits = 0

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is not None

    def get_ucb1(self):
        # children_values = [c.get_value() for c in self.children]
        # return np.mean(children_values)
        C = 2
        value = 0
        if self.visits == 0:
            value = 99999999999  # Infinite to avoid division by 0
        else:
            # Return ucb1 score = vi + c * sqrt(log(N)/ni)
            value = self.value / self.visits + C * \
                np.sqrt(np.log(self.parent.visits) / self.visits)
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

    def select(self, agent, max_iters=200):
        """ Explores and selects the best next state to choose.
        """
        current_node = self.root
        i = 0
        while(i < max_iters):
            i += 1
            print(f"Searching... Iter {i} of {max_iters}")
            if current_node.is_leaf:
                if current_node.visits == 0:  # Is new
                    res = self.simulate(current_node, agent)
                    self.backprop(current_node, res)
                else:  # Is not new
                    self.expand(current_node)
                    res = current_node.value
                    try:
                        max_child = np.argmax([x.value for x in
                                               current_node.get_ucb1()])
                        current_node = current_node.children[max_child]
                        res = self.simulate(current_node, agent)
                    except ValueError:  # If there were no children
                        pass
                    self.backprop(current_node, res)

                    # Return to root (finish cycle)
                    current_node = self.root
            else:
                ucbs = [u.get_ucb1() for u in current_node.children]
                best_child = np.argmax(ucbs)
                current_node = current_node.children[best_child]

        # Select the highest confidence one.
        max_val = np.argmax([v.get_ucb1() for v in self.root.children])
        # After the last play the oponent has played, so we use the before last
        return self.root.children[max_val].state.board.move_stack[-2]

    def expand(self, node):
        legal_moves = [m for m in node.state.get_legal_moves()]
        new_states = []
        for m in legal_moves:
            new_state = node.state.get_copy()
            new_state.move(m)
            new_states.append(new_state)

        node.children = [Node(s, parent=node) for s in new_states]

    def simulate(self, node, agent):
        """ Rollout from the current node until a final state. """
        sim_pol = RandomMovePolicy()
        sim = Simulation(agent, node.state.get_copy(), sim_pol)
        results_sim = sim.run(max_moves=200)  # []
        # TODO: N simulations
        # results_sim.append(sim.run(max_moves=100))
        return results_sim

    def backprop(self, node, value):
        node.visits += 1
        node.value += value

        if node.parent is not None:
            self.backprop(node.parent, value)
