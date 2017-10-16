# Gerasimos Lampouras, 2017:
from Action import Action

'''
 Each ActionSequence consists of an ArrayList of Actions.
 The ActionSequence typically begins with a series of content actions,
 and followed by subsets of word actions, each corresponding to one of the preceding content actions.
'''
class ActionSequence(object):

    def __init__(self, sequence, cleanEndTokens=False):
        self.sequence = []

        if isinstance(sequence, self.__class__):
            copySeq = sequence.sequence
        elif isinstance(sequence, list.__class__):
            copySeq = sequence

        for action in copySeq:
            # @cleanEndTokens: Whether or not Action.TOKEN_START and Action.TOKEN_END actions should be omitted.
            if not cleanEndTokens or (action.word != Action.TOKEN_START and action.word != Action.TOKEN_END):
                newAction = Action()
                newAction.copy(action)
                self.sequence.append(newAction)

    '''
     Replace the action at indexed cell of the ActionSequence, and shorten the sequence up to and including the index.
     Initially, this method is used to replace the action at timestep of a roll-in sequence with an alternative action.
     Afterwards, it shortens the sequence so that the rest of it (after index) be recalculated by performing roll-out.
    '''
    def modifyAndShortenSequence(self, index, modification):
        modification.copyAttrValueTracking(self.sequence[index])

        self.sequence.set(index, modification)
        self.sequence = self.sequence[:index + 1]

    '''
     Returns a string representation of the word actions in the ActionSequence.
    '''
    def getWordSequenceToString(self):
        w = str()
        for action in self.sequence:
            if action.word != Action.TOKEN_START and action.word != Action.TOKEN_END:
                w += action.word + " "
        return w.strip()

    '''    
     Returns a string representation of the word actions in the ActionSequence, while omitting all punctuation.
    '''
    def getWordSequenceToString_NoPunct(self):
        w = str()
        for action in self.sequence:
            if action.word != Action.TOKEN_START and action.word != Action.TOKEN_END and action.word != Action.TOKEN_PUNCT:
                w += action.word + " "
        return w.strip()

    '''
     Returns a string representation of the content actions in the ActionSequence.
    '''
    def getAttrSequenceToString(self):
        w = str()
        for action in self.sequence:
            w += action.attribute + " "
        return w.strip()

    '''   
     Returns the length of the sequence when not accounting for Action.TOKEN_START and Action.TOKEN_END actions.
    '''
    def getLength_NoBorderTokens(self):
        length = 0
        for action in self.sequence:
            if action.word != Action.TOKEN_START and action.word != Action.TOKEN_END:
                length += 1
        return length

    '''    
     Returns the length of the sequence when not accounting for Action.TOKEN_START and Action.TOKEN_END actions, 
     and punctuation.
    '''
    def getLength_NoBorderTokens_NoPunct(self):
        length = 0
        for action in self.sequence:
            if action.word != Action.TOKEN_START and action.word != Action.TOKEN_END and action.word != Action.TOKEN_PUNCT:
                length += 1
        return length

    '''
     Returns a subsequence consisting only of the content actions in the ActionSequence.
    '''
    def getAttributeSequence(self):
        attrSeq = []
        for action in self.sequence:
            attrSeq.append(action.attribute)
        return attrSeq

    '''
     Returns a subsequence consisting only of the content actions in the ActionSequence, up to a specified index.
    '''
    def getAttributeSubSequence(self, index):
        attrSeq = []
        for action in self.sequence[:index]:
            attrSeq.append(action.attribute)
        return attrSeq

    def __str__(self):
        return "ActionSequence{" + "sequence=" + self.sequence.__str__() + '}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.sequence.__str__() == other.sequence.__str__()
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __gt__(self, other):
        return len(self.sequence) > len(other.sequence)

    def __ge__(self, other):
        return (len(self.sequence) > len(other.sequence)) or (len(self.sequence) == len(other.sequence))

    def __lt__(self, other):
        return len(self.sequence) < len(other.sequence)

    def __le__(self, other):
        return (len(self.sequence) < len(other.sequence)) or (len(self.sequence) == len(other.sequence))

    def __hash__(self):
        return hash(self.getWordSequenceToString())
