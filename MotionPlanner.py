import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class MotionPlanner(nn.Module):

    def __init__(self, x_dim=1, hid_dim=128, z_dim=128, stop_threshold=0.):
        super().__init__()

        self.encoder = nn.Sequential(
            conv_block(x_dim, 64),
            conv_block(64, 32),
            conv_block(32, 16),
            nn.Flatten(),
            nn.Linear(76800, z_dim)
        )

        # Feed-forward, regression (single scalar)
        self.confPredictor = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

        # Feed-forward, regression (predicts a z_dim vector)
        self.transitionPredictor = nn.Sequential(
            nn.Linear(z_dim+1, hid_dim),            # The +1 is because we also concatenate an action to perform a transition
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim)
        )

        self.transitionPredictor.double()
        self.confPredictor.double()

        self.EMBEDDING_DIM = z_dim
        self.STOP_THRESHOLD = stop_threshold

    # 2 loss-terms, one that validates the predicted confidences, and one that enforces consistency between
    # the encoder and the predicted transitions, by making sure that each predicted transition matches the actual
    # encoded image at t+1.
    def calculate_loss(self, pred_conf, conf, pred_emb, enc_emb):
        # confidence_loss = F.smooth_l1_loss(pred_conf, conf) # pair-wise mean-squared error
        # transition_loss = F.smooth_l1_loss(pred_emb, enc_emb) # pair-wise mean_squared error
        confidence_loss = F.mse_loss(pred_conf.double(), conf.double()) # pair-wise mean-squared error
        transition_loss = F.mse_loss(pred_emb.double(), enc_emb.double()) # pair-wise mean_squared error

        print("confidence_loss = %s, transition_loss = %s" % (confidence_loss, transition_loss))
        return confidence_loss.double() + transition_loss.double()

    def trainSequence(self, images, actions, confidences):

        encoded_embeddings = self.encoder(images.float())
        pred_embeddings = []
        pred_confidences = []

        for i in range(len(images)):
            img_embedding = encoded_embeddings[i]

            if i > 0:
                # previous image embedding + current action ==> predict current image embedding
                prev_embedding = encoded_embeddings[i-1]
                # Note: a good alternative to just concatenating the action to the embedding would be to condition
                # the transition neural network on the action via FiLM layers (see BabyAI code for an example).

                prev_embedding = torch.reshape(prev_embedding, [1, -1])
                act = torch.reshape(torch.from_numpy(np.array([actions[i]])), [1, 1])
                transitionInput = torch.cat([prev_embedding, act], axis=-1)
                pred_embedding = self.transitionPredictor(transitionInput)
            else:
                pred_embedding = torch.reshape(img_embedding, [1, -1])

            pred_embeddings.append(pred_embedding)

            pred_conf = self.confPredictor(pred_embedding.double())
            pred_confidences.append(pred_conf)

        pred_confidences = torch.cat(pred_confidences)
        confidences = torch.from_numpy(np.array(confidences))
        pred_embeddings = torch.cat(pred_embeddings, axis=0)

        return self.calculate_loss(pred_confidences, confidences, pred_embeddings, encoded_embeddings)

    def evalConf(self, img):
        encoded_img = self.encoder(img.double())
        return self.confPredictor(encoded_img)

    def forward(self, img, a):
        # prediction of transition
        img = img.float()
        embedding = self.encoder(img)
        prev_embedding = torch.reshape(embedding, [1, -1])
        act = torch.reshape(torch.from_numpy(np.array([a])), [1, 1])
        transitionInput = torch.cat([prev_embedding, act], axis=-1)

        pred_embedding = self.transitionPredictor(transitionInput.double())

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

        gains = np.array([gain1.data.numpy(), gain2.data.numpy(), gain3.data.numpy(), gain4.data.numpy()])
        best_gain_idx = np.argmax(gains)
        best_gain = gains[best_gain_idx]

        # return action of best gain in value relative to current. If all relative gains are below a certain
        #  threshold, return action 0 (stop moving)
        if best_gain <= self.STOP_THRESHOLD:
            return 0, gains
        else:
            return best_gain_idx+1, gains
