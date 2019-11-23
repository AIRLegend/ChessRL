"""
Module with all the necessary stuff to encode/decode game states for using them
with a neural network. Also contains a Sequence Generator for fitting a
DatasetGame to a Model.
"""

import numpy as np
import chess

from tensorflow.keras.utils import Sequence, to_categorical


def _get_pieces_one_hot(board, color=False):
    """ Returns a 3D-matrix representation of the pieces for one color.
    The matrix ins constructed as follows:
        8x8 (Chess board) x 7 possible pieces (including empty). = 448.

    Parameters:
        board: Python-Chess Board. Board.
        color: Boolean, True for white, False for black
    Returns:
        mask: numpy array, 3D matrix with the pieces of the player.
    """
    mask = np.zeros((8, 8, 7))
    for i in list(chess.PIECE_TYPES):
        mask[:, :, i] = np.array(board.pieces(i, color).mirror()
                                 .tolist()).reshape(8, 8)
    # Encode blank positions
    mask[:, :, 0] = (~np.array(mask.sum(axis=-1), dtype=bool)).astype(int)
    return mask


def _get_current_game_state(board):
    """ This method returns the matrix representation of a game turn
    (positions of the pieces of the two colors)

    Parameters:
        board: Python-Chess board
    Returns:
        current: numpy array. 3D Matrix with dimensions 14x8x8.
        """
    return np.concatenate((_get_pieces_one_hot(board, color=False),
                           _get_pieces_one_hot(board, color=True)),
                          axis=-1)


def _get_game_history(board, T=8):
    """ Returns the matrix representation of a board history. If a game has no
    history, those turns will be considered null (represented as 0's matrices).

    Parameters:
        board: Python-Chess board (with or without moves)
        T: number of backwards steps to represent. (default 8 as in AlphaZero).
    Returns:
        history: NumPy array of dimensions 8x8x(14*T). Note that this history
        does not include the current game state, only the previous ones.
    """
    board_copy = board.copy()
    history = np.zeros((8, 8, 14 * T))

    for i in range(T):
        try:
            board_copy.pop()
        except IndexError:
            break
        history[:, :, i * 14: (i + 1) * 14] =\
            _get_current_game_state(board_copy)

    return history


def get_game_state(game, flipped=False):
    """ This method returns the matrix representation of a game with its
    history of moves.

    Parameters:
        game: Game. Game state.
    Returns:
        current: numpy array. 3D Matrix with dimensions 8x8x[14(T+1)]. Where T
        corresponds to the number of backward turns in time.
    """

    board = game.board
    current = _get_current_game_state(board)
    history = _get_game_history(board)
    current_turn = np.full((8, 8, 1), game.turn, dtype=float)  # Curr. turn
    current = np.concatenate((current, history, current_turn), axis=-1)

    if flipped:
        current = np.rot90(current, k=2)
    return current


def get_uci_labels():
    """ Returns a list of possible moves encoded as UCI (including
    promotions).
    Source:
        https://github.com/Zeta36/chess-alpha-zero/blob/
        master/src/chess_zero/config.py#L88
    """
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    promoted_to = ['q', 'r', 'b', 'n']

    for l1 in range(8):
        for n1 in range(8):
            destinations = [(t, n1) for t in range(8)] + \
                [(l1, t) for t in range(8)] + \
                [(l1 + t, n1 + t) for t in range(-7, 8)] + \
                [(l1 + t, n1 - t) for t in range(-7, 8)] + \
                [(l1 + a, n1 + b) for (a, b) in
                    [(-2, -1), (-1, -2), (-2, 1), (1, -2),
                     (2, -1), (-1, 2), (2, 1), (1, 2)]]

            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(8) and n2 in range(8):  # noqa: E501
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]  # noqa: E501
                    labels_array.append(move)

    for l1 in range(8):
        letter = letters[l1]
        for p in promoted_to:
            labels_array.append(letter + '2' + letter + '1' + p)
            labels_array.append(letter + '7' + letter + '8' + p)
            if l1 > 0:
                l_l = letters[l1 - 1]
                labels_array.append(letter + '2' + l_l + '1' + p)
                labels_array.append(letter + '7' + l_l + '8' + p)
            if l1 < 7:
                l_r = letters[l1 + 1]
                labels_array.append(letter + '2' + l_r + '1' + p)
                labels_array.append(letter + '7' + l_r + '8' + p)
    return labels_array


class DataGameSequence(Sequence):
    """ Transforms a Dataset to a Data generator to be fed to the training
    loop of the neural network.

    Attributes:
        dataset: DatasetGame. Dataset of fames
        batch_size: int. Nb of board representations of each batch
        uci_ids: dict. Encoding the move UCI labels to one-hot.
        random_flips: float. Proportion of board representation which will
                        be flipped 180 degrees.
    """

    def __init__(self, dataset: 'DatasetGame', batch_size: int = 8,  # noqa:F821
                 random_flips=0):
        self.dataset = dataset
        self.batch_size = min(batch_size, len(dataset))
        self.uci_ids = {u: i for i, u in enumerate(get_uci_labels())}
        self.random_flips = random_flips

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size:
                             (idx + 1) * self.batch_size]
        batch_x = []  # Board reprs
        batch_y_policies = []
        batch_y_values = []

        for i in batch:
            i_augmented = self.dataset.augment_game(i)

            flip = np.random.rand() < self.random_flips
            batch_x.extend([get_game_state(i_g['game'], flipped=flip)
                            for i_g in i_augmented])
            batch_y_policies.extend([
                to_categorical(self.uci_ids[targets['next_move']],
                               num_classes=1968)
                for targets in i_augmented]
            )
            batch_y_values.extend([targets['result']
                                   for targets in i_augmented])

        return np.asarray(batch_x), (np.asarray(batch_y_policies),
                                     np.asarray(batch_y_values))
