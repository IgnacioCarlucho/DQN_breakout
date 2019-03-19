# DQN

Re implementation of original DQN paper for playing in atari environments. 

![](myimage.gif)

## How to run

Runs the default environment breakout. 
```
python main.py 
```

You can add different arguments to run the agent:
```
python main.py --epsilon 0.1 --load True
```
Runs the agent using the pretrained model (trained on 2M frames) with episolon lower.