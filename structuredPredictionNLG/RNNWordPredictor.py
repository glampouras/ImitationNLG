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

        self.softmax = nn.LogSoftmax()

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

        return (self.softmax(logits), (new_hidden, new_cell))


class RNNWordPredictor(object):
    """Needs to implement sklearn model interface, i.e. .fit() and .predict()
    """
    def __init__(self, vocab_size, embed_size, hidden_size):

        self.model = RNNWordPredictorModel(vocab_size, embed_size, hidden_size)
        self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def predict(self, input_word_index, hidden_state):
        # The features should be at least a tuple of (input word, state)
        # state itself is a tuple of (hidden state, cell state)
        log_probs, new_state = self.model(Variable(torch.LongTensor([input_word_index])), hidden_state)

        return (log_probs, new_state)

    def fit(self, action_costs, encoded_labels):
        # training_data: features to make a prediction
        # encoded_labels: what the correct prediction should be

        # Actually, all we need to record as the feature structure is the
        # word and the hidden state, as they suffice to make the prediction

        # Although the hidden state is only a summary of the words so far
        # GIVEN THE CURRENT PARAMETERS, so we might have to aggregate updates
        # for everything in training_data - worries about batch size?

        # It's fine if we assume structuredInstances is a batch of training data

        loss = 0
        counter = 0
        self.optimizer.zero_grad()
        for action_cost, label in zip(action_costs, encoded_labels):
            loss += self.loss(action_cost, Variable(torch.LongTensor([label])))
            counter += 1

        loss /= counter
        loss.backward()
        self.optimizer.step()

        return loss
