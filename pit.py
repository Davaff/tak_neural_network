import numpy as np

import Arena
from MCTS import MCTS
from Tak import Tak
from TakNN import TakNN

g = Tak()
print(g.getActionSize())

n1 = TakNN()
n2 = TakNN()
n2.loadWeights("rejected_weights")

mcts1 = MCTS(n1)
mcts2 = MCTS(n2)

n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

print(Arena.Arena(n1p, n2p).playGames(2, verbose=False))
