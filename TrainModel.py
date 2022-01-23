from MotionPlanner import MotionPlanner
import torch.optim as optim

NUM_EPOCHS = 1000
LR = 0.0001

# TODO: load data: each sequence is a sequence of images combined with a sequence of preceding actions and a
#  confidence value associated with each image

#  N = number of total sequences to train on
#  T = number of timesteps in the sequences: this can vary from sequence to sequence, hence we have embedded lists,
#   but not numpy arrays (or tensors, either)

X = []      # shape = [N, T, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS]
a = []      # shape = [N, T, 1]
c = []      # shape = [N, T, 1]

model = MotionPlanner()
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):

    optimizer.zero_grad()

    epoch_loss = 0.
    for i in range(len(X)):
        # TODO: batching

        loss = model.trainSequence(X[i], a[i], c[i])

        epoch_loss += loss

    optimizer.zero_grad()
    epoch_loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    # TODO: after N epochs, eval on the validation set

    print("Epoch# %s: loss = %s" % (epoch+1, epoch_loss.data.numpy()))

    # TODO: save best model to file

