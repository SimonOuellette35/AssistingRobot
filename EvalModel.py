from MotionPlanner import MotionPlanner
import torch
import utils
import numpy as np
import matplotlib.pyplot as plt

WIDTH = 640
HEIGHT = 480
NUM_CHANNELS = 3
data_dir = './data/training_data/'

VIZ_CONF_MODEL = False
VIZ_TRANS_MODEL = True

# load model
model = MotionPlanner(x_dim=NUM_CHANNELS)
model.load_state_dict(torch.load('best_model'))
model.eval()

# load the eval data
X, a, c = utils.loadData(data_dir)

if VIZ_CONF_MODEL:
    # 1) evaluate the confidence estimation module, by visualizing randomly sampled images and their estimated
    #  confidences
    for _ in range(10):
        random_test_idx = np.random.choice(np.arange(len(X)))
        random_step_idx = np.random.choice(np.arange(len(X[random_test_idx])))

        img = np.reshape(X[random_test_idx][random_step_idx], (1, NUM_CHANNELS, HEIGHT, WIDTH))
        pred_conf = model.evalConf(img)

        img = np.reshape(X[random_test_idx][random_step_idx], (HEIGHT, WIDTH, NUM_CHANNELS))
        plt.imshow(img)
        plt.title("Confidence ground truth: %s, Predicted confidence: %s" % (
            c[random_test_idx][random_step_idx][0],
            pred_conf.data.numpy()[0][0]))
        plt.show()

if VIZ_TRANS_MODEL:
    # 2) evaluate the planning module, by visualizing randomly sampled images and the projected confidences associated
    #  with each direction of movement.
    accuracy = 0.
    for _ in range(10):
        random_test_idx = np.random.choice(np.arange(len(X)))
        random_step_idx = np.random.choice(np.arange(len(X[random_test_idx])))

        img = np.reshape(X[random_test_idx][random_step_idx], (1, NUM_CHANNELS, HEIGHT, WIDTH))
        pred, gains = model.predictBestAction(img, 0.)

        img = np.reshape(X[random_test_idx][random_step_idx], (HEIGHT, WIDTH, NUM_CHANNELS))
        if gains[0][0][0] > gains[1][0][0]:
            print("predicted action = clockwise, ground truth = ", a[random_test_idx][random_step_idx+1])
        else:
            print("predicted action = counter-clockwise, ground truth = ", a[random_test_idx][random_step_idx+1])

        plt.imshow(img)
        plt.title("Clockwise: %s, counter-clockwise: %s" % (
            gains[0][0][0],
            gains[1][0][0]
        ))
        plt.show()

# TODO: it seems that right now it only moves left-ward, even when moving right-ward would get us to a higher confidence
#  position much faster.
#  a) it suggests that the transition module doesn't correctly transform the embedding into one that reduces
#  confidence. It seems to just have simply learned that in general, moving left-ward increases confidence?? Review
#  training loop. Is it learning to match the confidence of the predicted image, or the real one???
#  b) Would some loss term or planning objective to minimize movements (with offline multi-step
#  planning, rather than greedy next-step prediction) solve this?
#
# # evaluate imagined trajectories
# for test_idx in range(len(a)):
#
#     step = 0
#     done = False
#
#     # what is the estimated value of X[step]?
#     current_img = X[test_idx][step]
#     current_img = current_img.unsqueeze(0)
#     current_conf = model.evalConf(current_img)
#
#     while not done:
#
#         best_a, gains = model.predictBestAction(current_img, current_conf)
#
#         print("Test #%s, step %s: current confidence = %s, best action = %s (ground truth = %s)" % (
#             test_idx,
#             step,
#             current_conf.data.numpy(),
#             best_a,
#             a[test_idx][step+1]
#         ))
#
#         if step >= len(a[test_idx]) - 2:
#             done = True
#
#         if current_conf >= 0.8:
#             print("Stopping motion planning, confidence threshold reached!")
#             done = True
#
#         step += 1
#
#     current_img = X[test_idx][step]
#     current_img = current_img.unsqueeze(0)
#     current_conf = model.evalConf(current_img)
#
#     print("Ending confidence = ", current_conf.data.numpy())