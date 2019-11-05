# Neural chess
## Reinforcement Learning based chess engine.
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

### Requirements
The necessary python packages (ATM) are listed in the requirements file.
Install them with
> pip3 install -r requirements.txt

Also, you need to download the specific 
[stockfish binary](https://stockfishchess.org/download/) for your platform, put it
under the `res/stockfish/stockfish-10-64` path and make it executable with `chmod +x
res/stockfish/stockfish-10-64`

**OPTIONAL**: Apart from those packages, you should have Graphviz installed in your computer if
you want to visualize the Monte Carlo trees.

### Training
> **DISCLAIMER:** This is under development and still contains bugs

You can start a training job with
```bash
python src/chessrl/training.py ../../data/models/model0 --games=4 --threads=2
```

This will create 2 threads which will play 4 games generating a dataset of moves, and then, a model will be trained and saved under `../../data/models/model0` directory. If there is a model already saved there, the script will train tnat existing model with the new data.

### Literature

1. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning
   Algorithm, Silver, D. et al. [https://arxiv.org/pdf/1712.01815.pdf]
2. Mastering the game of Go without human knowledge. Silver, D. et al. [https://www.nature.com/articles/nature24270.epdf?author_access_token=VJXbVjaSHxFoctQQ4p2k4tRgN0jAjWel9jnR3ZoTv0PVW4gB86EEpGqTRDtpIz-2rmo8-KG06gqVobU5NSCFeHILHcVFUeMsbvwS-lxjqQGg98faovwjxeTUgZAUMnRQ]

