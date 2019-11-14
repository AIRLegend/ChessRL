# Neural chess <br> Reinforcement Learning based chess engine.

Personal project to build a chess engine based using reinforcement learning.

The idea is to some sort replicate the system built by DeepMind with AlphaZero. I'm
aware that the computational resources to achieve their results is huge, but my aim
it's simply to reach an amateur chess level performance (about 1200-1400 Elo), not
state of the art.

At this moment, the approach I'm using is similar to the original work but changing
the AI playing vs itself with Stockfish as the opponent, which is the best engine
available (the one which was beaten by AlphaZero). This way, I suppose it will
converge faster towards an "acceptable" game style than following the AlphaZero way.

*Work in progress*

## Requirements
The necessary python packages (ATM) are listed in the requirements file.
Install them with

```bash
pip3 install -r requirements.txt
```

Also, you need to download the specific 
[stockfish binary](https://stockfishchess.org/download/) for your platform,
for automating this made a script to automatically download it.

```bash
cd res
chmod +x get_stockfish.sh
./get_stockfish.sh linux   #or "mac", depending of your platform. 
```
If you want to download it manually, you have to put the stockfish executable under `res/stockfish-10-64` path, in order to the training script to detect it.

>**OPTIONAL**: Apart from those packages, you should have Graphviz installed in your computer if
you want to visualize the Monte Carlo trees.

## Training
> **DISCLAIMER:** This is under development and can still contains bugs or  inefficiencies

You can start a training job with

```bash
cd src/chessrl
python training.py ../../data/models/model0 --workers=2 --train_rounds 10
```
This will create 2 child processes which will play 2 concurrent games, generating a dataset of moves, and then, a model will be trained and saved under `../../data/models/model0` 
directory (repeating that for 10 rounds). If there is a model already saved there, 
the script will train that existing model with the new data.

The script will use a GPU if available, if not, the CPU.

To list all the available options run

```bash
python training.py --help
```

## How do I view the progress?

The neural network trainning evolution can be monitored with Tensorboard, simply:

```bash
tensorboard --logdir data/models/model0/train
```
(And set the "Horizontal axis" to "WALL" for viewing all the diferent runs.)

Also, in the same model directory you will find a `gameplays.json` file which
contains the recorded training games of the model. With this, we can study its
behaviour over time.

## Can I play against the agent?

Yup. Under `src/webplayer` you will find a Flask app which deploys a web interface to play against the trained agent. There is another README with more information.


## Literature

1. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
   Algorithm, Silver, D. et al. https://arxiv.org/pdf/1712.01815.pdf
2. Mastering the game of Go without human knowledge. Silver, D. et al. https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ

