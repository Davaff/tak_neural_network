import os
import time

import numpy as np
from keras import Input, Model
from keras.layers import Activation, BatchNormalization, Conv2D, Add, Flatten, Dropout, Dense
from keras.optimizers import Adam

class TakNN:

    def __init__(self):
        print("##### Building neural network #####")

        self.size = 5
        self.max_height = 43
        self.batchSize = 64
        self.epochs = 100

        self.input_layer = Input(shape=(self.size, self.size, self.max_height))
        # layer 1: convolution to 256 channels.
        layer1 = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(self.input_layer)))
        # layer 2: ResNet block 1
        # Every block contains two rectified batch-normalized convolutional layers with a skip connection.
        layer2_a = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer1)))
        layer2_b = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer2_a)))
        layer2 = Activation("relu")(Add()([layer2_b, layer1]))

        # layer 3: ResNet block 2
        layer3_a = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer2)))
        layer3_b = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer3_a)))
        layer3 = Activation("relu")(Add()([layer3_b, layer2]))

        # layer 4: ResNet block 3
        layer4_a = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer3)))
        layer4_b = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer4_a)))
        layer4 = Dropout(0.3)(Activation("relu")(Add()([layer4_b, layer3])))  # Dropout in training to prevent
        # over fitting.

        head_split = Activation("relu")(BatchNormalization(axis=3)(Conv2D(1, 1, padding="same")(layer4)))
        head_flat = Flatten()(head_split)
        head_2 = Activation("relu")(BatchNormalization(axis=1)(Dense(256)(head_flat)))
        self.v = Dense(1, activation="tanh", name="v")(head_2)

        policy_split = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 1, padding="same")(layer4)))
        policy_2 = Activation("relu")(
            BatchNormalization(axis=3)(Conv2D(151, 1, padding="same", use_bias=True)(policy_split)))
        policy_flat = Flatten()(policy_2)
        self.pi = Dense(3775, activation="softmax", name="pi")(policy_flat)

        self.model = Model(inputs=self.input_layer, outputs=[self.v, self.pi])
        self.model.compile(loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(0.001))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(x=input_boards, y=[target_pis, target_vs], batch_size=self.batchSize, epochs=self.epochs)

    def predict(self, board):
        start = time.time()
        board = board[np.newaxis, :, :]
        pi, v = self.model.predict(board, verbose=False)

        print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path '{}'".format(filepath))
        self.model.load_weights(filepath)
