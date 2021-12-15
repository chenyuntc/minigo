# Learning to Play Go
This repository contains the code of our CSC2515 research project.

In the project, we implement an agent that learns to play Go on 9*9 board through behaviour cloning from human knowledge and then refined with self-play. Our agent convincingly beats all baselines including several famous Go programs. 

Table of Contents
=================
  * [Install Dependancy](#install-dependancy)
  * [Prepare Data](#prepare-data)
  * [Training](#training)
  * [Run Engine on Command Line](#run-engine-on-command-line)
  * [Run Engine on Sabaki](#run-engine-on-sabaki)

## Install Dependancy
You need to install following packages in order to run the code:
- [PyTorch>=1.3.1](https://pytorch.org/)
- tqdm
- dill
- fire

## Prepare Data

Unzip the sgf data. 
``` python
unzip -q sgf.zip
```

## Training
``` python
python train_behaviour_cloning.py --steps=1280000
```

Several parameters can be specified:

| Parameter        | Default |                                           |
| ---------------- | ------- | ----------------------------------------- |
| data_dir      | "sgf"   | Path to the data directory.               |
| steps         | 400000  | Training steps.                           |
| verbose_step  | 1000    | Print verbose.                            |
| batch_size    | 2048    | Batch size. Recommend to be at least 128. |
| learning_rate | 1e-3    | Learning rate.                            |

## Run Engine on Command Line

With the model that just trained, we can now deploy it with [GTP](https://www.gnu.org/software/gnugo/gnugo_19.html) to play with other Go programes. 

We can start our pre-trained engine by:

```bash
chmod 777 dlgo.py
./dlgo.py --weights weights-name --playouts 1600 --resign-threshold 0.25
```

Several parameters can be specified:

| Parameter          | Default |                                                              |
| ------------------ | ------- | ------------------------------------------------------------ |
| weights          |         | Path to the pre-trained model.                               |
| playouts         | 400     | MCTS playouts,                                               |
| resign-threshold | 0.1     | If the possibility of win is smaller than the specified value, then resign. |

You can interactively play it though GTP, e.g. `genmove black` will generate a move for black. 

## Run Engine on Sabaki

Since the code supported GTP, it can be intergrated with other Go GUI like [Sabaki](https://github.com/SabakiHQ/Sabaki), an elegant Go board and SGF editor. Go engines can be added to Sabaki to play offline. Sabaki then acts as a graphical UI for any Go software that supports [GTP (Go Text Protocol)](https://www.lysator.liu.se/~gunnar/gtp/).

Step1: open manage engine. 

<img src="./img/sabaki_01.png" style="zoom:50%;" />

Step 2, add new engine. 

<img src="./img/sabaki_02.png" style="zoom:50%;" />

Step 3, load the engine. 

<img src="./img/sabaki_03.png" style="zoom:50%;" />

<img src="./img/sabaki_04.png" style="zoom:50%;" />



## Self-play training

**Step 1**:  First we need to run selected some game position to run MCTS. It will take about 10 minutes, to dump 100K game positions.

```python
python dump_position_from_datasets.py
```

**Step 2:** Run MCTS on these selected positions, this step is very slow. If you want to run it on distributed, you need to install [horovod](https://github.com/horovod/horovod),

```python
python -m horovod.runner.launch -np 16 python dump_self_play_results.py
```

It takes ~24 hours in a 2 1080ti machines to dump the results.

**Step 3**: Run refinement on the results. This step takes about 4 hours.

```bash
python train_self_play.py
```

## code ownership

We borrowed a lot of code from other repos. For those python files that were borrowed/referred from other repo, we added a comment on top of the `.py` files. Generally speaking, we refer a lot to [CGLemon/pyDLGO](https://github.com/CGLemon/pyDLGO/).   Here is the description of some key files that are copied from other repos:

1. `board.py` for Go board operation/representation, is copied  from [ymgaq/Pyaq](https://github.com/ymgaq/Pyaq)
2.  `gtp.py`  for support of basic GTP , copied from [CGLemon/pyDLGO](https://github.com/CGLemon/pyDLGO/), 
3.  `mcts.py` Monte Carlo tree search,  copied from [CGLemon/pyDLGO](https://github.com/CGLemon/pyDLGO/)
4.  `sgf.py` Python implementation of Smart Game Format (SGF) to read human game record,  copied from  [jtauber/sgf](https://github.com/jtauber/sgf)
5. `network.py`, network architecture, based on [CGLemon/pyDLGO](https://github.com/CGLemon/pyDLGO/), but with a lot of simplification and cleaning.
6. `train_self_play.py`, `train_behaviour_cloning.py`, `dataset.py`, `dump_self_play_data.py`  and `dump_position_from_datasets` are written by us.