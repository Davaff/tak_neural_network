from TakNN import TakNN
from Trainer import Trainer

examples = "examples_1"
nn = TakNN()
trainer = Trainer(nn)
trainer.deleteExamples(examples)
trainer.generateExamples(self_play=False)

