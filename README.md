# Detector optimizer

At the current  moment only a Time of flight detector is present

## Components:
 - SKRL
 - pythorch

## Modules
- **env.py**: Contains environment definition and behaviour, multpile subclasses are present for different step behaviour
- **misc.py**: Contains helper functions utilized by the environment, also contains the definition of reward calculator class and reward methods and simulation of detector
- **PPO.ipynb**: Contains implementation and training of **PPO** algorith described in https://arxiv.org/abs/1707.06347
- **RPO.ipynb**: Contains implementation and training of **RPO** algorith described in https://arxiv.org/abs/2212.07536
- **DDPG.ipynb**: Contains implementation and training of **DDPG** algorith described in https://arxiv.org/abs/1509.02971
- **SAC.ipynb**: Contains implementation and training of **SAC** algorith described in https://arxiv.org/abs/1801.01290



