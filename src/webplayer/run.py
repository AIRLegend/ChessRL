from flask import Flask, render_template, request, make_response, jsonify

import sys
import os

def process_initializer():
    """ Initializer of the training threads in in order to detect if there
    is a GPU available and use it. This is needed to initialize TF inside the
    child process memory space."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

process_initializer()


sys.path.append('../chessrl/')

import gamewrapper  # noqa:E402

app = Flask("Chess")


# Init instance
_ = gamewrapper.GameWrapper.get_instance()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/game/<move>', methods=['POST'])
def user_move(move):
    g = gamewrapper.GameWrapper.get_instance()
    g.move(move)
    response = jsonify({'fen': g.fen, 'hist': g.get_history()})
    return make_response(response, 200)


@app.route("/game", methods=['GET', 'DELETE'])
def get_game_state():
    if request.method == "GET":
        g = gamewrapper.GameWrapper.get_instance()
        response = jsonify({'fen': g.fen, 'hist': g.get_history()})
        return response
    elif request.method == "DELETE":
        gamewrapper.GameWrapper.destroy_instance()
        return "Deleted", 200


@app.route("/game/<color>", methods=['PUT'])
def change_color(color):
    whites = True
    if color == "0":
        whites = False
    gamewrapper.GameWrapper.destroy_instance()

    g = gamewrapper.GameWrapper.get_instance(player_color=whites)
    # The user is blacks, so we force the game start by pushing a null move
    if not whites:
        g.move('null move')

    return "Changed", 200


if __name__ == "__main__":
    app.run('0.0.0.0', debug=True, processes=1, threaded=False)
