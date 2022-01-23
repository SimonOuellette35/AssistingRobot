import torch.nn.functional as F
import numpy as np

class MotionPlanner:

    def __init__(self):
        self.encoder = None # TODO
        self.confPredictor = None # TODO
        self.transitionPredictor = None # TODO
        self.STOP_THRESHOLD = 0.

    # 2 loss-terms, one that validates the predicted confidences, and one that enforces consistency between
    # the encoder and the predicted transitions, by making sure that each predicted transitions matche the actual
    # encoded images at t+1.
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
                pred_embedding = self.transitionPredictor(img_embedding, actions[i+1])  # TODO: a(t+1) vs image(t-1)???
            else:
                pred_embedding = img_embedding

            pred_embeddings.append(pred_embedding)

            pred_conf = self.confPredictor(pred_embedding)
            pred_confidences.append(pred_conf)

        return self.calculate_loss(pred_confidences, confidences, pred_embeddings, encoded_embeddings)

    def predict(self, img, a):
        # prediction of transition
        pred_embedding = self.transitionPredictor(img, a)

        # return predicted value for predicted transition
        return self.confPredictor(pred_embedding)

    def predictBestAction(self, img, current_value):
        pred1 = self.predict(img, 1)
        pred2 = self.predict(img, 2)
        pred3 = self.predict(img, 3)
        pred4 = self.predict(img, 4)

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

