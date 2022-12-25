import os
import time
import numpy as np
from keras import Input, Model
from keras.layers import Activation, BatchNormalization, Conv2D, Add, Flatten, Dropout, Dense
from keras.optimizers import Adam

from Tak import Tak


# The class for the neural network.
# Weights can be saved or retrieved from the "weights" folder.
class TakNN:
    def __init__(self):
        print("##### Building neural network #####")

        self.size = 5
        self.max_height = 43
        self.batchSize = 64
        self.epochs = 100
        self.resNetBlocks = 10

        self.input_layer = Input(shape=(self.size, self.size, self.max_height))
        # layer 1: convolution to 256 channels.
        layer1 = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(self.input_layer)))

        # Every res net block contains two rectified batch-normalized convolutional layers with a skip connection.
        # Creating the res net blocks
        old_in = layer1
        for i in range(0, self.resNetBlocks-1):
            old_in = self.resNetBlock(old_in)

        # Last res net block with some dropout to avoid overfiitting
        layer4_a = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(old_in)))
        layer4_b = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer4_a)))
        layer4 = Dropout(0.3)(Activation("relu")(Add()([layer4_b, old_in])))
        # over fitting.

        head_split = Activation("relu")(BatchNormalization(axis=3)(Conv2D(1, 1, padding="same")(layer4)))
        head_flat = Flatten()(head_split)
        head_2 = Activation("relu")(BatchNormalization(axis=1)(Dense(256)(head_flat)))
        self.v = Dense(1, activation="tanh", name="v")(head_2)

        policy_split = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 1, padding="same")(layer4)))
        policy_2 = Activation("relu")(
            BatchNormalization(axis=3)(Conv2D(Tak().actions_per_field, 1, padding="same", use_bias=True)(policy_split)))
        policy_flat = Flatten()(policy_2)
        self.pi = Dense(Tak.getActionSize(), activation="softmax", name="pi")(policy_flat)

        self.model = Model(inputs=self.input_layer, outputs=[self.v, self.pi])
        self.model.compile(loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(0.001))
    def resNetBlock(self, input_layer):
        layer2_a = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(input_layer)))
        layer2_b = Activation("relu")(BatchNormalization(axis=3)(Conv2D(256, 3, padding="same")(layer2_a)))
        return Activation("relu")(Add()([layer2_b, input_layer]))
    def train(self, examples):
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.model.fit(x=input_boards, y=[target_vs, target_pis], batch_size=self.batchSize, epochs=self.epochs)

    def predict(self, board):
        start = time.time()
        board = board[np.newaxis, :, :]
        v, pi = self.model.predict(board, verbose=False)
        #print("Prediction took: {0:03f} seconds.".format(time.time() - start))
        return pi[0], v[0]

    def saveWeights(self, title: str):
        filepath = os.path.join("./weights/", title)
        if not os.path.exists("./weights/"):
            os.mkdir("./weights/")
        self.model.save_weights(filepath, save_format="tf")

    def loadWeights(self, title: str):
        filepath = os.path.join("./weights/", title)
        self.model.load_weights(filepath)
