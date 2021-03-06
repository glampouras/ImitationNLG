# Gerasimos Lampouras, 2017:
from copy import copy
import imitation

'''
 Action implements a single action of the NLG action sequences.
 The same structure is used for both content and word actions.
'''
class Action(object):
    # Default action identifier for attribute shifting in an NLGState.
    TOKEN_EOS = "@eos@"
    # Default action identifier for attribute shifting in an NLGState.
    TOKEN_SHIFT = "@shift@"
    # Default action identifier for punctuation occurrences in an NLGState.
    TOKEN_PUNCT = "@punct@"
    # Default action identifier for variable occurrences in an NLGState.
    # This is usually combined with an attribute identifier and an index integer to distinguish between variables.
    TOKEN_X = "@x@"
    # Default action identifier to start decoding
    TOKEN_GO = "@go@"

    def __init__(self, word, attribute):
        # Main components of an action
        # In practice, attribute may be set as "attribute|value"
        # Each action consists of a word and corresponding attribute to which the word aligns to.
        self.label = word
        self.attribute = attribute

    def __str__(self):
        return "A{" + self.label + "," + self.attribute + "}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.label == other.label
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.label)
