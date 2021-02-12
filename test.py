import random
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim

from utils import accuracy, preprocess, get_images_from


# hyperparameters
ways=2; shots=1; meta_lr=0.003; fast_lr=0.5; meta_batch_size=32; adaptation_steps=1; cuda=False; seed=42

# load saved model
model = l2l.vision.models.OmniglotFC(28 ** 2, ways)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
loss = nn.CrossEntropyLoss(reduction='mean')

# acquire adaptation dataset
folder='./adaptset/'
image0 = preprocess(folder+'0.jpg')
image1 = preprocess(folder+'1.jpg')
adaptation_data = torch.cat([image0, image1], dim=0)
adaptation_labels = torch.tensor([0, 1])

# Adapt the model
train_error = loss(maml(adaptation_data), adaptation_labels)
maml.adapt(train_error)

# acquire evaluation dataset
evaluation_data = get_images_from(folder+'testset/')

# Evaluate the adapted model
predictions = maml(evaluation_data)
print('predictions:', predictions.argmax(dim=1).view(-1))
