from Tak import Tak, MoveAction, Direction
from TakNN import TakNN
import numpy as np

from Trainer import Trainer

# Credit to https://github.com/suragnair/alpha-zero-general
import sys
sys.setrecursionlimit(10000)
nn = TakNN()
c = Trainer(nn)
c.train()

