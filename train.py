from TakNN import TakNN
from Trainer import Trainer

nn = TakNN()
trainer = Trainer(nn)
#trainer.generateExamples("examples_1", False)
trainer.loadTrainExamples("examples_1")
trainer.train()
