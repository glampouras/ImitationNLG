import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RNNNLGState(object):
    def __init__(self):
        self.actions_taken = []

    def updateWithAction(self, action, structuredInstance):
        # Here, action is a word. Add it to the list of words generated
        self.actions_taken.append(action)


class RNNWordPredictorModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(RNNWordPredictorModel, self).__init__()

        self.input_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        self.softmax = nn.Softmax()

    def forward(self, input, state):
        '''
        Args:
            input: word index
            state: state at previous timestep

        Returns:
            log_probs: log_probabilites of subsequent word
            new_state: updated state
        '''

        # Embed the input word
        input_embed = self.input_embeddings(input)

        # Calculate the new hidden state
        new_hidden, new_cell = self.lstm_cell(input_embed, state)

        # Calculate the log-likelihoods of the next action
        logits = self.output_projection(new_hidden)

        return (logits, (new_hidden, new_cell))


class RNNWordPredictor(object):
    """Needs to implement sklearn model interface, i.e. .fit() and .predict()
    """
    def __init__(self, vocab_size, embed_size, hidden_size):

        self.model = RNNWordPredictorModel(vocab_size, embed_size, hidden_size)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def predict(self, input_word_index, hidden_state):
        # The features should be at least a tuple of (input word, state)
        # state itself is a tuple of (hidden state, cell state)
        log_probs, new_state = self.model(Variable(torch.LongTensor([input_word_index])), hidden_state)

        return (log_probs, new_state)

    def fit(self, training_data, encoded_labels):
        # training_data: features to make a prediction
        # encoded_labels: what the correct prediction should be

        # Actually, all we need to record as the feature structure is the
        # word and the hidden state, as they suffice to make the prediction

        # Although the hidden state is only a summary of the words so far
        # GIVEN THE CURRENT PARAMETERS, so we might have to aggregate updates
        # for everything in training_data - worries about batch size?

        # It's fine if we assume structuredInstances is a batch of training data

        predictions = []
        for data, label in zip(training_data, encoded_labels):
            # Work out what the model prediction is
            prediction, _ = self.predict(data)
            predictions.append(prediction)

        # Calculate the cost of the prediction
        loss = nn.CrossEntropyLoss()
        cost = sum(map(loss, zip(predictions, encoded_labels)))

        loss.backward()
