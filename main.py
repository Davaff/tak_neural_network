import tensorflow

from Tak import Tak, MoveAction, Direction
from TakNN import TakNN
import numpy as np

from Trainer import Trainer

# Credit to https://github.com/suragnair/alpha-zero-general
import sys

print(tensorflow.test.gpu_device_name())
sys.setrecursionlimit(10000)

nn = TakNN()
#nn.loadWeights(f"new_curr_weights3")
c = Trainer(nn)
while True:
    c.generateExamples(write_to_file=False, self_play=False)
    c.train()

