from MotionPlanner import MotionPlanner
import torch.optim as optim
import numpy as np
import torch
import utils

NUM_EPOCHS = 2500
LR = 0.0001
data_dir = './data/training_data/'
NUM_CHANNELS = 3

X, a, c = utils.loadData(data_dir)
model = MotionPlanner(x_dim=NUM_CHANNELS)
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)

best_loss = np.inf
for epoch in range(NUM_EPOCHS):

    optimizer.zero_grad()

    epoch_loss = 0.
    for i in range(len(X)):
        loss = model.trainSequence(X[i], a[i], c[i])

        epoch_loss += loss

    optimizer.zero_grad()
    epoch_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    # TODO: after N epochs, eval on the validation set

    print("Epoch# %s: loss = %s" % (epoch+1, epoch_loss.data.numpy()))

    # save best model to file
    if epoch_loss.data.numpy() < best_loss:
        best_loss = epoch_loss.data.numpy()

        torch.save(model.state_dict(), 'best_model')