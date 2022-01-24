from MotionPlanner import MotionPlanner
import torch.optim as optim
import os
import csv
import numpy as np
import cv2
import torchvision.transforms as transforms
import torch

NUM_EPOCHS = 1000
LR = 0.0001
data_dir = './data/training_data/'

#  N = number of total sequences to train on
#  T = number of timesteps in the sequences: this can vary from sequence to sequence, hence we have embedded lists,
#   but not numpy arrays (or tensors, either)

X = []      # shape = [N, T, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS]
a = []      # shape = [N, T, 1]
c = []      # shape = [N, T, 1]

# Load data: each sequence is a sequence of images combined with a sequence of preceding actions and a
# confidence value associated with each image.
directory = os.fsencode(data_dir)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        action_sequence = []
        with open("%s%s" % (data_dir, filename), 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')

            for a in datareader:
                action_sequence.append(float(a[0]))

        a.append(np.reshape(action_sequence, [-1, 1]))
    elif filename.endswith(".avi"):
        image_sequence = []

        # load video frame by frame
        cap = cv2.VideoCapture('%s%s' % (data_dir, filename))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame_count += 1

                # convert OpenCV image to array or tensor
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Define a transform to convert the image to tensor
                transform = transforms.ToTensor()

                # Convert the image to PyTorch tensor
                tensor = transform(image)
                image_sequence.append(tensor)
            else:
                cap.release()

        cap.release()
        cv2.destroyAllWindows()

        X.append(image_sequence)

        conf_sequence = []
        for i in range(frame_count):
            tmp = (i / (frame_count-1))
            conf_sequence.append(tmp)

        c.append(np.reshape(conf_sequence, [-1, 1]))

model = MotionPlanner()
model.train()
optimizer = optim.AdamW(model.parameters(), lr=LR)

best_loss = np.inf
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

    # save best model to file
    if epoch_loss.data.numpy() < best_loss:
        best_loss = epoch_loss.data.numpy()

        torch.save(model.state_dict(), 'best_model')