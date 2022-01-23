import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class MotionPlanner(nn.Module):

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64, stop_threshold=0.):
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        # Feed-forward, regression (single scalar)
        self.confPredictor = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

        # Feed-forward, regression (predicts a z_dim vector)
        self.transitionPredictor = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm2d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim)
        )

        self.EMBEDDING_DIM = z_dim
        self.STOP_THRESHOLD = stop_threshold

    # 2 loss-terms, one that validates the predicted confidences, and one that enforces consistency between
    # the encoder and the predicted transitions, by making sure that each predicted transition matches the actual
    # encoded image at t+1.
    def calculate_loss(self, pred_conf, conf, pred_emb, enc_emb):
        confidence_loss = F.smooth_l1_loss(pred_conf, conf) # pair-wise mean-squared error
        transition_loss = F.smooth_l1_loss(pred_emb, enc_emb) # pair-wise mean_squared error

        return confidence_loss + transition_loss

    def trainSequence(self, images, actions, confidences):

        encoded_embeddings = []
        pred_embeddings = []
        pred_confidences = []

        for i in range(len(images)):
            img_embedding = self.encoder(images[i])
            encoded_embeddings.append(img_embedding)

            if i > 0:
                # previous image embedding + current action ==> predict current image embedding
                prev_embedding = self.encoder(images[i-1])
                pred_embedding = self.transitionPredictor(prev_embedding, actions[i])
            else:
                pred_embedding = img_embedding

            pred_embeddings.append(pred_embedding)

            pred_conf = self.confPredictor(pred_embedding)
            pred_confidences.append(pred_conf)

        return self.calculate_loss(pred_confidences, confidences, pred_embeddings, encoded_embeddings)

    def forward(self, img, a):
        # prediction of transition
        pred_embedding = self.transitionPredictor(img, a)

        # return predicted value for predicted transition
        return self.confPredictor(pred_embedding)

    def predictBestAction(self, img, current_value):
        pred1 = self.forward(img, 1)
        pred2 = self.forward(img, 2)
        pred3 = self.forward(img, 3)
        pred4 = self.forward(img, 4)

        gain1 = pred1 - current_value
        gain2 = pred2 - current_value
        gain3 = pred3 - current_value
        gain4 = pred4 - current_value

        gains = np.array([gain1, gain2, gain3, gain4])
        best_gain_idx = np.argmax(gains)
        best_gain = gains[best_gain_idx]

        # return action of best gain in value relative to current. If all relative gains are below a certain
        #  threshold, return action 0 (stop moving)
        if best_gain <= self.STOP_THRESHOLD:
            return 0
        else:
            return best_gain_idx+1

