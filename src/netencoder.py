"""
Module with all the necessary stuff to encode/decode game states for using them
with a neural network
"""

import numpy as np
import chess


def get_pieces_one_hot(board, color=False):
    """ Returns a 3D-matrix representation of the pieces for one color.
    The matrix ins constructed as follows:
        8x8 (Chess board) x 7 possible pieces (including empty). = 448.

    Parameters:
        board: Python-Chess Board. Board.
        color: Boolean, True for white, False for black
    Returns:
        mask: numpy array, 3D matrix with the pieces of the player.
    """
    mask = np.zeros((7, 8, 8))
    for i in list(chess.PIECE_TYPES):
        mask[i, :, :] = np.array(board.pieces(i, color).mirror()
                                 .tolist()).reshape(8, 8)
        # Encode blank positions
        mask[0, :, :] = (~np.array(mask.sum(axis=0), dtype=bool)).astype(int)
        return mask


def get_current_game_state(board):
    """ This method returns the matrix representation of a game turn
    (positions of the pieces of the two colors)

    Returns:
        current: numpy array. 3D Matrix with dimensions 14x8x8.
        """

    return np.concatenate((get_pieces_one_hot(board, color=False),
                           get_pieces_one_hot(board, color=True)),
                          axis=0)


def get_game_history(board, T=8):
    board_copy = board.copy()
    history = np.zeros((14 * T, 8, 8))

    for i in range(T):
        try:
            board_copy.pop()
        except IndexError:
            break
        history[i * 14: (i + 1) * 14, :, :] =\
            get_current_game_state(board_copy)

    return history


def get_game_state(board):
    """ This method returns the matrix representation of a game with its
    history of moves.

    Returns:
        current: numpy array. 3D Matrix with dimensions (14T)x8x8. Where T
        corresponds to the number of backward turns in time. (And the 14 is
        the current representation of the two players) """

    current = get_current_game_state(board)
    history = get_game_history(board)
    current = np.concatenate((current, history), axis=0)
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
