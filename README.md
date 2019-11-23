# Neural chess <br> Reinforcement Learning based chess engine.

Personal project to build a chess engine based using reinforcement learning.

The idea is to some sort replicate the system built by DeepMind with AlphaZero. I'm
aware that the computational resources to achieve their results is huge, but my aim
it's simply to reach an amateur chess level performance (about 1200-1400 Elo), not
state of the art.

At this moment, the approach I'm using is based on pre-training a model using self-play data of a Stockfish algorithm. Later, the idea is to put two models to play agaisnt each other and make selection/merges of weights (RL part).

*Work in progress*

## Requirements
The necessary python packages (ATM) are listed in the requirements file.
You can install them with

```bash
pip3 install -r requirements.txt
```

Tensorflow is also needed, but you must install either `tensorflow`or `tensorflow-gpu`.

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
> **DISCLAIMER:** This is under development and can still contains bugs or  inefficiencies and modifications are being made.

> **DISCLAIMER 2:** As soon as I get acceptable results, I will also share weights/datasets with this code.

For training an agent in a supervised way you will need a saved dataset of games. The script `gen_data_stockfish.py`is made for generating a JSON with this. This script will play (and record) several games using two Stockfish instances. Execute it first to create this dataset (take a look at it's possible arguments)

Then, start the supervised training:

```bash
cd src/chessrl
python supervised.py ../../data/models/model1 ../../data/dataset_stockfish.json --epochs 2 --bs 4
```

**TODO**
Unsupervised stage.

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

