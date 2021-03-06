import matplotlib.pyplot as plt

import random
import numpy as np
import torch
import learn2learn as l2l

from torch import nn, optim

# from torch.utils.tensorboard import SummaryWriter



def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


ways=2 # 5
shots=1
meta_lr=0.003
fast_lr=0.5
meta_batch_size=32
adaptation_steps=1
num_iterations=600 # 60000
cuda=True
seed=42


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cpu')
if cuda:
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda')

# Load train/validation/test tasksets using the benchmark interface
tasksets = l2l.vision.benchmarks.get_tasksets('omniglot',
                                              train_ways=ways,
                                              train_samples=2*shots,
                                              test_ways=ways,
                                              test_samples=2*shots,
                                              num_tasks=20000,
                                              root='~/data',
)

# Create model
model = l2l.vision.models.OmniglotCNN(28 ** 2, ways) # OmniglotFC
model.to(device)
maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
opt = optim.Adam(maml.parameters(), meta_lr)
loss = nn.CrossEntropyLoss(reduction='mean')



# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('runs/meta_l_omniglot')

# for iteration in range(num_iterations):
for iteration in range(num_iterations, 10*num_iterations):
    opt.zero_grad()

    meta_train_error = 0
    meta_train_accuracy = 0
    meta_valid_error = 0
    meta_valid_accuracy = 0

    for task in range(meta_batch_size):
        # Compute meta-training loss
        learner = maml.clone()
        batch = tasksets.train.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                            learner,
                                                            loss,
                                                            adaptation_steps,
                                                            shots,
                                                            ways,
                                                            device)
        evaluation_error.backward()
        meta_train_error += evaluation_error.item()
        meta_train_accuracy += evaluation_accuracy.item()

        # Compute meta-validation loss
        learner = maml.clone()
        batch = tasksets.validation.sample()
        evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                            learner,
                                                            loss,
                                                            adaptation_steps,
                                                            shots,
                                                            ways,
                                                            device)
        meta_valid_error += evaluation_error.item()
        meta_valid_accuracy += evaluation_accuracy.item()

    # writer.add_scalar('meta_train_error', meta_train_error / meta_batch_size, iteration)
    # writer.add_scalar('meta_train_accuracy', meta_train_accuracy / meta_batch_size, iteration)
    # writer.add_scalar('meta_valid_error', meta_valid_error / meta_batch_size, iteration)
    # writer.add_scalar('meta_valid_accuracy', meta_valid_accuracy / meta_batch_size, iteration)

    # Average the accumulated gradients and optimize
    for p in maml.parameters():
        p.grad.data.mul_(1.0 / meta_batch_size)
    opt.step()

meta_test_error = 0.0
meta_test_accuracy = 0.0
for task in range(meta_batch_size):
    # Compute meta-testing loss
    learner = maml.clone()
    batch = tasksets.test.sample()
    evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                        learner,
                                                        loss,
                                                        adaptation_steps,
                                                        shots,
                                                        ways,
                                                        device)
    meta_test_error += evaluation_error.item()
    meta_test_accuracy += evaluation_accuracy.item()
print('Meta Test Error', meta_test_error / meta_batch_size)
print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size)


torch.save(model.state_dict(), 'model.pth')