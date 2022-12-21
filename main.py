from TakNN import TakNN
import numpy as np

# Credit to https://github.com/suragnair/alpha-zero-general

nn = TakNN()

test_board = np.random.rand(5, 5, 43)
test_board = np.expand_dims(test_board, axis=0)

prediction = nn.model.predict(test_board, verbose=False)
print(prediction[0])
print(prediction[1])
