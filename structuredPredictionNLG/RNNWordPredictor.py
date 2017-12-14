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
    def __init__(self, vocab_size, attr_vocab_size, val_vocab_size,
                 embedding_size, hidden_size):
        super(RNNWordPredictor, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.attr_embeddings = nn.Embedding(attr_vocab_size, embedding_size)
        self.val_embeddings = nn.Embedding(val_vocab_size, embedding_size)
        self.lstm_cell = nn.LSTMCell(embedding_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        self.softmax = nn.LogSoftmax()

        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, word, attr, val, state):
        '''
        Args:
            word: word index
            attr: attribute index
            val: the value of the attribute (if applicable)
            state: state at previous timestep

        Returns:
            log_probs: log_probabilites of subsequent word
            new_state: updated state
        '''

        word_var = Variable(torch.LongTensor(np.array([word])))
        attr_var = Variable(torch.LongTensor(np.array([attr])))
        val_var = Variable(torch.LongTensor(np.array([val])))

        if self.is_cuda:
            word_var = word_var.cuda()
            attr_var = attr_var.cuda()
            val_var = val_var.cuda()

        # Embed the inputs
        word_embed = self.word_embeddings(word_var)
        attr_embed = self.attr_embeddings(attr_var)
        val_embed = self.val_embeddings(val_var)

        embed = word_embed + attr_embed + val_embed

        # Calculate the new hidden state
        new_hidden, new_cell = self.lstm_cell(embed, state)

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
