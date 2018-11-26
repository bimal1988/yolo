# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datasets import YOLODataset
from utils.parse_config import parse_model_config
from utils.utils import weights_init_normal
from models import Darknet

cuda = torch.cuda.is_available()

model_config_path = "config/yolov3.cfg"
train_path = "data/test.txt"

# Get hyperparameters
hyperparams = parse_model_config(model_config_path)[0]
learning_rate = float(hyperparams["learning_rate"])
momentum = float(hyperparams["momentum"])
decay = float(hyperparams["decay"])
burn_in = int(hyperparams["burn_in"])

# Initiate model
model = Darknet(model_config_path)
model.apply(weights_init_normal)

model.train()

# Get dataloader
dataloader = DataLoader(YOLODataset(train_path), batch_size=1, shuffle=False)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()))
epochs = 30
for epoch in range(epochs):
    for minibatch, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)

        optimizer.zero_grad()

        loss = model(imgs, targets)

        loss.backward()
        optimizer.step()

        print(
            "[Epoch %d/%d, Batch %d/%d]"
            "[Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, "
            "recall: %.5f, precision: %.5f]"
            % (
                epoch,
                epochs,
                minibatch,
                len(dataloader),
                model.losses["x"],
                model.losses["y"],
                model.losses["w"],
                model.losses["h"],
                model.losses["conf"],
                model.losses["cls"],
                loss.item(),
                model.losses["recall"],
                model.losses["precision"],
            )
        )
