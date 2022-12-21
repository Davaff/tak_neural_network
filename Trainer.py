from collections import deque
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
from Tak import Tak
from TakNN import TakNN


def getCheckpointFile(iteration):
    return 'checkpoint_' + str(iteration) + '.pth.tar'


class Trainer:

    def __init__(self, net):
        self.updateThreshold = 0.55
        self.numItersForTrainExamplesHistory = 20
        self.selfPlayEpisodes = 100
        self.trainExamplesHistory = []
        self.trainingIterations = 10
        self.neuralNetwork: TakNN = net
        self.competitorNetwork: TakNN = TakNN()  # Competitor
        self.mcts: MCTS = MCTS(self.neuralNetwork)
        self.arenaCompare = 40
        self.checkpoint = "./checkpoint/"
        self.game = Tak()
        self.tempThreshold = 30

    def train(self):
        for i in range(1, self.trainingIterations + 1):
            print(f"##### Starting iteration {i} #####")

            if i > 1:
                iterationTrainExamples = deque([], maxlen=200000)

                for _ in tqdm(range(self.selfPlayEpisodes), desc="Self Play"):
                    self.mcts = MCTS(self.neuralNetwork)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.neuralNetwork.save_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            self.competitorNetwork.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.competitorNetwork)

            self.neuralNetwork.train(trainExamples)
            nmcts = MCTS(self.neuralNetwork)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)))
            pwins, nwins, draws = arena.playGames(self.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.updateThreshold:
                print('REJECTING NEW MODEL')
                self.neuralNetwork.load_checkpoint(folder=self.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.neuralNetwork.save_checkpoint(folder=self.checkpoint, filename=getCheckpointFile(i))
                self.neuralNetwork.save_checkpoint(folder=self.checkpoint, filename='best.pth.tar')

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

