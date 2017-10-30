[![Build Status](https://travis-ci.org/pfackeldey/RL-with-Keras.svg?branch=master)](https://travis-ci.org/pfackeldey/RL-with-Keras)

# RL-with-Keras

Reinforced Learning with Keras

To install all necessary packages run:

```sh
sudo pip install -r requirements.txt
```

Execute training with: `python dqn.py --train`! Use argument `--gpu`, if you want to train with your GPU (Default: all CPU cores). If you want to see the development of the performance while training, use the argument `--watch-agent-train` (CAUTION: This option slows down training!).

In order to visualize training do: `tensorboard --logdir summary/Breakout-v0` and open the link in your browser.
