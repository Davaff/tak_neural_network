from collections import deque
from random import shuffle
import numpy as np
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
from Tak import Tak, PlaceAction
from TakNN import TakNN


class Trainer:

    def __init__(self, net):
        self.updateThreshold = 0.55
        self.numItersForTrainExamplesHistory = 20
        self.selfPlayEpisodes = 50
        self.trainExamplesHistory = []
        self.trainingIterations = 30
        self.neuralNetwork: TakNN = net
        self.competitorNetwork: TakNN = TakNN()  # Competitor
        self.game = Tak()
        self.arenaCompare = 40
        self.tempThreshold = 15

    def train(self):
        for i in range(1, self.trainingIterations + 1):
            print(f"##### Starting iteration {i} #####")

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
            self.neuralNetwork.saveWeights(f"weight_iter_{i}")
            self.competitorNetwork.loadWeights(f"weight_iter_{i}")
            prev_mcts = MCTS(self.competitorNetwork)

            self.neuralNetwork.train(trainExamples)
            new_mcts = MCTS(self.neuralNetwork)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(prev_mcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(new_mcts.getActionProb(x, temp=0)))
            prev_wins, new_wins, draws = arena.playGames(self.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (new_wins, prev_wins, draws))
            if prev_wins + new_wins == 0 or float(new_wins) / (prev_wins + new_wins) < self.updateThreshold:
                print('REJECTING NEW MODEL')
                self.neuralNetwork.loadWeights(f"weight_iter_{i}")
            else:
                print('ACCEPTING NEW MODEL')
                self.neuralNetwork.saveWeights(f"weight_iter_{i}")

    def executeEpisode(self):
        """
        Executes one episode of self-play.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        currPlayer = 1
        episodeStep = 0

        steps = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, currPlayer)
            temp = int(episodeStep < self.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, currPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            steps += 1

            board, currPlayer = self.game.getNextState(board, currPlayer, action)

            r = self.game.getGameEnded(board, currPlayer)

            if steps > 1000 and r == 0:
                r = 0.001

            if r != 0:
                # Board, pi, v
                return [(x[0], x[2], r * ((-1) ** (x[1] != currPlayer))) for x in trainExamples]
