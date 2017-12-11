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


class RNNWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(RNNWordPredictor, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.input_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        self.softmax = nn.LogSoftmax()

        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, input, state):
        '''
        Args:
            input: word index
            state: state at previous timestep

        Returns:
            log_probs: log_probabilites of subsequent word
            new_state: updated state
        '''

        input_var = Variable(torch.LongTensor(np.array([input])))
        if self.is_cuda:
            input_var = input_var.cuda()

        # Embed the input word
        input_embed = self.input_embeddings(input_var)

        # Calculate the new hidden state
        new_hidden, new_cell = self.lstm_cell(input_embed, state)

        # Calculate the log-likelihoods of the next action
        logits = self.output_projection(new_hidden)

        return (self.softmax(logits), (new_hidden, new_cell))

    def init_hidden(self):
        if self.is_cuda:
            return (Variable(torch.zeros(1, self.hidden_size).cuda()),
                    Variable(torch.zeros(1, self.hidden_size).cuda()))
        else:
            return (Variable(torch.zeros(1, self.hidden_size)),
                    Variable(torch.zeros(1, self.hidden_size)))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def fit(self, probs, labels):
        loss = 0.0
        probs = torch.cat(probs, dim=0)
        labels = Variable(torch.LongTensor((list(labels))))
        if self.is_cuda:
            labels = labels.cuda()
        loss = self.loss(probs, labels)

        loss.backward()
        self.optimizer.step()

        return loss
