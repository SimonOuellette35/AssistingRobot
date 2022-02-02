import os
import csv
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

WIDTH = 640
HEIGHT = 480
NC = 3

def loadData(data_dir):

    #  N = number of total sequences to train on
    #  T = number of timesteps in the sequences: this can vary from sequence to sequence, hence we have embedded lists,
    #   but not numpy arrays (or tensors, either)

    X = []      # shape = [N, T, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS]
    a = []      # shape = [N, T, 1]
    c = []      # shape = [N, T, 1]

    # Load data: each sequence is a sequence of images combined with a sequence of preceding actions and a
    # confidence value associated with each image.
    directory = os.fsencode(data_dir)

    timestamps = []
    for file in os.listdir(directory):
        timestamp = os.path.splitext(file)[0].decode("utf-8")
        timestamp = timestamp.replace("_masked", "")
        if timestamp not in timestamps:
            timestamps.append(timestamp)

    print("Found timestamps: ", timestamps)

    for timestamp in timestamps:
        print("Loading timestamp %s" % timestamp)
        csv_filename = os.fsdecode("%s.csv" % timestamp)
        avi_filename = os.fsdecode("%s_masked.avi" % timestamp)

        action_sequence = []
        with open("%s%s" % (data_dir, csv_filename), 'r') as csvfile:
            datareader = csv.reader(csvfile, delimiter=',')

            for row in datareader:
                action_sequence.append(float(row[0]))

        tmp_a = np.reshape(action_sequence, [-1, 1])
        a.append(tmp_a)

        image_sequence = []

        # load video frame by frame
        cap = cv2.VideoCapture('%s%s' % (data_dir, avi_filename))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                frame_count += 1

                # convert OpenCV image to array or tensor
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # # Define a transform to convert the image to tensor
                # transform = transforms.ToTensor()

                # Convert the image to PyTorch tensor
                np_image = np.reshape(image, [1, NC, HEIGHT, WIDTH])
                image_sequence.append(torch.from_numpy(np_image))
            else:
                cap.release()

        cap.release()
        cv2.destroyAllWindows()

        img_sequence_tensor = torch.cat(image_sequence, axis=0)

        X.append(img_sequence_tensor)

        conf_sequence = []
        for i in range(frame_count):
            tmp = (float(i) / float(frame_count-1))
            conf_sequence.append(tmp)

        c.append(np.reshape(conf_sequence, [-1, 1]))

    return X, a, c