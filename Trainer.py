import os
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from keras.callbacks import EarlyStopping
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
from Tak import Tak, PlaceAction
from TakNN import TakNN


class Trainer:

    def __init__(self, net):
        self.updateThreshold = 0.55
        self.selfPlayEpisodes = 60
        self.trainExamples = []
        self.neuralNetwork: TakNN = net
        self.competitorNetwork: TakNN = TakNN()  # Competitor
        self.game = Tak()
        self.arenaCompare = 40
        self.tempThreshold = 15

    def generateExamples(self, name, write_to_file=False):
        for _ in tqdm(range(self.selfPlayEpisodes), desc="Self Play"):
            self.trainExamples += self.executeEpisode()
        print(f"Generated {len(self.trainExamples)} examples for training.")
        if write_to_file:
            self.saveTrainExamples(name)

    def train(self):
        print(f"##### Starting training #####")
        shuffle(self.trainExamples)
        self.neuralNetwork.saveWeights(f"old_weights")
        self.competitorNetwork.loadWeights(f"old_weights")
        prev_mcts = MCTS(self.competitorNetwork)

        self.neuralNetwork.train(self.trainExamples)
        new_mcts = MCTS(self.neuralNetwork)

        print('PITTING AGAINST PREVIOUS VERSION')
        arena = Arena(lambda x: np.argmax(prev_mcts.getActionProb(x, temp=0)),
                      lambda x: np.argmax(new_mcts.getActionProb(x, temp=0)))
        prev_wins, new_wins, draws = arena.playGames(self.arenaCompare)

        print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (new_wins, prev_wins, draws))
        if prev_wins + new_wins == 0 or float(new_wins) / (prev_wins + new_wins) < self.updateThreshold:
            print('REJECTING NEW MODEL')
            self.neuralNetwork.saveWeights("rejected_weights")
        else:
            print('ACCEPTING NEW MODEL')
            self.neuralNetwork.saveWeights("accepted_weights")

    def executeEpisode(self):
        """
        Executes one episode of self-play.
        """
        mcts = MCTS(self.neuralNetwork)  # reset search tree
        roundExamples = []
        board = self.game.getInitBoard()
        currPlayer = 1
        episodeStep = 0

        steps = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, currPlayer)
            temp = int(episodeStep < self.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                roundExamples.append([Tak.stringRepresentation(b), currPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            steps += 1

            board, currPlayer = self.game.getNextState(board, currPlayer, action)

            r = self.game.getGameEnded(board, currPlayer)

            if steps > 1000 and r == 0:
                r = 0.001

            if r != 0:
                # Board, pi, v
                return [(x[0], x[2], r * ((-1) ** (x[1] != currPlayer))) for x in roundExamples]

    def saveTrainExamples(self, name):
        folder = "./examples/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, name)
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamples)
        f.close()

    def loadTrainExamples(self, name):
        filename = os.path.join("./examples/", name)
        if not os.path.isfile(filename):
            raise "No file with examples found"
        else:
            print("##### File with trainExamples found. Loading it... #####")
            with open(filename, "rb") as f:
                print(f"File size {os.stat(filename).st_size}")
                self.trainExamples = Unpickler(f).load()
            print('Loading done!')
