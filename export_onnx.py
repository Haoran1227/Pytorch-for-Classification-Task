import torch as t
from trainer import Trainer
import sys
from model.resnet import Resnet

#epoch = int(sys.argv[1])
epoch = 4

model = Resnet()

crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
