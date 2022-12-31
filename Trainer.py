import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import numpy as np
from keras.callbacks import EarlyStopping
from pwnlib.tubes.process import process
from tqdm import tqdm
from Arena import Arena
from MCTS import MCTS
from Tak import Tak, PlaceAction
from TakNN import TakNN
from Utility import Stone


class Trainer:

    def __init__(self, net):
        sys.setrecursionlimit(10000)
        self.updateThreshold = 0.55
        self.selfPlayEpisodes = 1
        self.trainExamples = []
        self.neuralNetwork: TakNN = net
        self.competitorNetwork: TakNN = TakNN()  # Competitor
        self.arenaCompare = 20
        self.tempThreshold = 15

    def train(self):
        shuffle(self.trainExamples)
        self.neuralNetwork.train(self.trainExamples)
        self.neuralNetwork.saveWeights(f"new_curr_weights4")
        return
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
        board = Tak.getInitState()
        currPlayer = 1
        episodeStep = 0

        steps = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, currPlayer)
            temp = int(episodeStep < self.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            roundExamples.append([Tak.stringRepresentation(canonicalBoard), currPlayer, pi, None])

            action = np.random.choice(len(pi), p=pi)
            steps += 1

            board, currPlayer = self.game.getNextState(board, currPlayer, action)

            r = self.game.getGameEnded(board, currPlayer)

            if steps > 1000 and r == 0:
                r = 0.001

            if r != 0:
                # Board, pi, v
                return [(x[0], x[2], r * ((-1) ** (x[1] != currPlayer))) for x in roundExamples]

    def deleteExamples(self, name):
        folder = "./examples/"
        filename = os.path.join(folder, name)
        if os.path.exists(filename):
            os.remove(filename)

    def generateExamples(self, self_play=False, name="", write_to_file=False):
        for _ in tqdm(range(self.selfPlayEpisodes), desc="Self Play"):
            #try:
            ep = self.executeEpisode() if self_play else self.executeTakticianEpisode()
            self.trainExamples += ep
            #for board, pi, v in ep:
                #print(f"Pi is {pi}")
                #print(f"V is {v}")
                #Tak.printBoard(Tak.boardRepresentation(board))
            if write_to_file:
                self.saveTrainExamples(name, ep)
            #except:
                #print("Exception!")
        print(f"Generated {len(self.trainExamples)} examples for training.")

    def takticianWhite(self):
        io = process("./taktician play -size=5 -white=minimax:5", shell=True)

        mcts = MCTS(self.neuralNetwork)  # reset search tree
        roundExamples = []
        state = Tak.getInitState()
        episodeStep = 0

        canonicalBoard = Tak.getCanonicalForm(state)
        while True:
            episodeStep += 1
            temp = 0  # int(episodeStep < self.tempThreshold)

            print(io.recvuntil("stones:").decode("ascii"))
            io.recvline()
            tact_act = io.recvline().decode("ascii")[3:-1].strip()
            print(f"Action from taktician {tact_act}")
            print(io.recvline().decode("ascii"))
            # Tak.printBoard(state.board)
            action = Tak.parseTakticianResponse(tact_act)
            pi = [0] * Tak.getActionSize()
            pi[action.getActionInt()] = 1
            roundExamples.append([Tak.stringRepresentation(canonicalBoard.board), state.curr_player, pi, None])

            state = Tak.getNextState(state, action.getActionInt())
            # Tak.printBoard(state.board)
            canonicalBoard = Tak.getCanonicalForm(state)
            Tak.printBoard(canonicalBoard.board)
            score = self.neuralNetwork.predict(board=canonicalBoard.board)
            print(f"Score for prev canonical board: {score[1][0]}")

            r = Tak.getGameEnded(state)

            if r != 0:
                if r == 1:
                    print(f"Game won by player {-state.curr_player} after {episodeStep} steps.")
                elif r == -1:
                    print(f"Game lost by player {-state.curr_player} after {episodeStep} steps.")
                # Board, pi, v
                return [(x[0], x[2], r * ((-1) ** (x[1] == state.curr_player))) for x in roundExamples]

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            # val = Tak.getValidMoves(state)
            # val = [x / sum(val) for x in val]
            action = np.random.choice(len(pi), p=pi)

            # try:
            print(io.recvuntil("black>").decode("ascii"))
            # except EOFError:
            #    Tak.getGameEnded(state)
            ac = Tak.getAction(action).getTakticianCommand()
            print(f"Sending action {ac}")
            io.sendline(ac)
            state = Tak.getNextState(state, action)
            canonicalBoard = Tak.getCanonicalForm(state)
            Tak.printBoard(canonicalBoard.board)
            score = self.neuralNetwork.predict(board=canonicalBoard.board)
            print(f"Score for prev canonical board: {score[1][0]}")
            #roundExamples.append([Tak.stringRepresentation(canonicalBoard.board), state.curr_player, pi, None])

            r = Tak.getGameEnded(state)

            if r != 0:
                if r == 1:
                    print(f"Game won by player {-state.curr_player} after {episodeStep} steps.")
                elif r == -1:
                    print(f"Game lost by player {-state.curr_player} after {episodeStep} steps.")
                # Board, pi, v
                return [(x[0], x[2], r * ((-1) ** (x[1] == state.curr_player))) for x in roundExamples]

    def takticianBlack(self):
        io = process("./taktician play -size=5 -black=minimax:5", shell=True)

        mcts = MCTS(self.neuralNetwork)  # reset search tree
        roundExamples = []
        state = Tak.getInitState()
        episodeStep = 0

        canonicalBoard = Tak.getCanonicalForm(state)
        while True:
            episodeStep += 1
            temp = 0  # int(episodeStep < self.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)
            # val = Tak.getValidMoves(state)
            # val = [x / sum(val) for x in val]
            action = np.random.choice(len(pi), p=pi)

            # try:
            print(io.recvuntil("white>").decode("ascii"))
            # except EOFError:
            #    Tak.getGameEnded(state)
            ac = Tak.getAction(action).getTakticianCommand()
            print(f"Sending action {ac}")
            io.sendline(ac)
            state = Tak.getNextState(state, action)
            canonicalBoard = Tak.getCanonicalForm(state)
            Tak.printBoard(canonicalBoard.board)
            score = self.neuralNetwork.predict(board=canonicalBoard.board)
            print(f"Score for prev canonical board: {score[1][0]}")
            #roundExamples.append([Tak.stringRepresentation(canonicalBoard.board), state.curr_player, pi, None])

            print(io.recvuntil("stones:").decode("ascii"))
            io.recvline()
            tact_act = io.recvline().decode("ascii")[7:-1].strip()
            print(f"Action from taktician {tact_act}")
            print(io.recvline().decode("ascii"))
            # Tak.printBoard(state.board)

            action = Tak.parseTakticianResponse(tact_act)
            pi = [0] * Tak.getActionSize()
            pi[action.getActionInt()] = 1
            roundExamples.append([Tak.stringRepresentation(canonicalBoard.board), state.curr_player, pi, None])

            state = Tak.getNextState(state, action.getActionInt())
            # Tak.printBoard(state.board)
            canonicalBoard = Tak.getCanonicalForm(state)
            Tak.printBoard(canonicalBoard.board)
            score = self.neuralNetwork.predict(board=canonicalBoard.board)
            print(f"Score for prev canonical board: {score[1][0]}")

            r = Tak.getGameEnded(state)

            if r != 0:
                if r == 1:
                    print(f"Game won by player {-state.curr_player} after {episodeStep} steps.")
                elif r == -1:
                    print(f"Game lost by player {-state.curr_player} after {episodeStep} steps.")
                # Board, pi, v
                return [(x[0], x[2], r * ((-1) ** (x[1] == state.curr_player))) for x in roundExamples]

    def executeTakticianEpisode(self):
        """
        Executes one episode of play against the taktician AI.
        """
        return self.takticianBlack() + self.takticianWhite()

    def saveTrainExamples(self, name, example):
        folder = "./examples/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, name)
        with open(filename, "ab+") as f:
            for ex in example:
                Pickler(f).dump(ex)
        f.close()

    def loadTrainExamples(self, name):
        filename = os.path.join("./examples/", name)
        if not os.path.isfile(filename):
            raise "No file with examples found"
        else:
            print("##### File with trainExamples found. Loading it... #####")
            with open(filename, "rb") as f:
                print(f"File size {os.stat(filename).st_size}")
                examples = 0
                while True:
                    try:
                        self.trainExamples.append(Unpickler(f).load())
                        examples += 1
                    except EOFError:
                        break
                print(f"Imported {examples} examples")
            print('Loading done!')
