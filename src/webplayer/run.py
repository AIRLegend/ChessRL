from flask import Flask, render_template, request, make_response, jsonify

import sys
sys.path.append('/Users/air/Documents/Projects/neuralchess/src/chessrl')

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
    app.run(debug=True, processes=1, threaded=False)
