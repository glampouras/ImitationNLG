# Gerasimos Lampouras, 2017:
from Action import Action
import imitation
import copy
from collections import deque
from collections import defaultdict
import torch
from torch.autograd import Variable

'''
 Each NLGState consists of an ArrayList of Actions.
 The NLGState is typically initialized with a series of content actions,
 and is populated with word actions, each corresponding to one of the preceding content actions.
'''
class NLGState(imitation.State):

    def __init__(self, contentPredictor, datasetInstance, useExpertContent=False):
        self.actionsTaken = []
        self.tokensProduced = []
        self.RNNState = []
        self.actionProbsCache = []
        self.expertActions = []
        self.expertActionsTaken = []
        self.datasetInstance = datasetInstance
        if datasetInstance:
            if useExpertContent:
                self.agenda = deque(self.optimalContentPolicy())
            else:
                self.agenda = deque(contentPredictor.rollContentSequence_withLearnedPolicy(datasetInstance))

            # Shift first attribute
            # self.actionsTaken.append(Action(Action.TOKEN_SHIFT, self.agenda[0][0]))
            # Add a @go@ symbol to initialise decoding
            self.tokensProduced.append(Action(Action.TOKEN_GO, self.agenda[0][0]))
            # TODO: where to put this?
            self.RNNState.append(
                (Variable(torch.zeros(1, 100)),
                 Variable(torch.zeros(1, 100))))
        '''
        if isinstance(sequence, self.__class__):
            copySeq = sequence.actionsTaken
        else:
            copySeq = sequence

        for action in copySeq:
            # @cleanEndTokens: Whether or not Action.TOKEN_SHIFT and Action.TOKEN_EOS actions should be omitted.
            if not cleanShiftTokens or (action.label != Action.TOKEN_SHIFT and action.label != Action.TOKEN_EOS):
                newAction = Action()
                newAction.copy(action)
                self.actionsTaken.append(newAction)
        '''

    '''
     Clone constructor.
     @param other NLGState whose values will be used to instantiate this object
    '''
    def clone(self, other):
        self.actionsTaken = copy.deepcopy(other.actionsTaken)
        self.tokensProduced = copy.deepcopy(other.tokensProduced)
        self.RNNState = copy.copy(other.RNNState)
        self.actionProbsCache = copy.copy(other.actionProbsCache)
        self.expertActions = copy.deepcopy(other.expertActions)
        self.expertActionsTaken = copy.deepcopy(other.expertActionsTaken)
        self.datasetInstance = copy.deepcopy(other.datasetInstance)
        self.agenda = copy.deepcopy(other.agenda)

    # extract features for current action in the agenda
    # probably not useful to us
    def extractFeatures(self, mrl, action):
        pass

    def getRNNFeatures(self):
        return self.tokensProduced[-1], self.RNNState[-1]

    def optimalContentPolicy(self):
        seq = [o.attribute for o in self.datasetInstance.directReferenceSequence if o.label == Action.TOKEN_SHIFT]
        return list([(attr, self.datasetInstance.input.attributeValues[attr]) for attr in seq])

    def optimalPolicy(self):
        availableWords = set()
        seqLen = len([o for o in self.actionsTaken if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS])
        isSeqLongerThanAllRefs = True
        for indirectRef in self.datasetInstance.output.evaluationReferenceActionSequences:
            if seqLen < len(indirectRef):
                isSeqLongerThanAllRefs = False
            for action in indirectRef:
                availableWords.add(action.label)
        costVector = defaultdict(lambda: 1.0)

        # If the sequence is longer than all the available references, it has gone on too far and should stop
        if isSeqLongerThanAllRefs:
            costVector[Action.TOKEN_EOS] = 0.0
        else:
            rollIn = [o.label.lower() for o in self.actionsTaken if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS]
            for ref in self.datasetInstance.output.evaluationReferenceActionSequences:
                for i in range(0, len(ref)):
                    # Do not repeat the same word twice in a row
                    if not self.actionsTaken or ref[i].label != self.actionsTaken[-1].label.lower():
                        if ref[i].label == Action.TOKEN_EOS:
                            rollOutSeq = rollIn[:]
                            refCost = self.datasetInstance.output.compareAgainst(rollOutSeq)
                            if refCost.loss < costVector[Action.TOKEN_EOS]:
                                costVector[Action.TOKEN_EOS] = refCost.loss
                        else:
                            rollOutSeq = rollIn[:]
                            if ref[-1].label == Action.TOKEN_EOS:
                                rollOutSeq.extend([o.label for o in ref[i:-1]])
                            else:
                                rollOutSeq.extend([o.label for o in ref[i:]])
                            refCost = self.datasetInstance.output.compareAgainst(rollOutSeq)

                            if refCost.loss < costVector[ref[i].label]:
                                costVector[ref[i].label] = refCost.loss
                            if ref[i].attribute != self.agenda[0][0]:
                                if refCost.loss < costVector[Action.TOKEN_SHIFT]:
                                    costVector[Action.TOKEN_SHIFT] = refCost.loss

        minCost = min(costVector.values())
        if minCost != 0.0:
            for word in costVector:
                costVector[word] = costVector[word] - minCost
        bestLabels = set([act for act in costVector if costVector[act] == 0.0])

        # This does allow subsequent SHIFT actions, with no words generated between them.
        # It might encourage learning to produce no words for some attributes, let's keep that in mind.
        if Action.TOKEN_EOS in bestLabels and Action.TOKEN_SHIFT in bestLabels:
            if len(self.agenda) == 1:
                return Action.TOKEN_EOS, costVector
            else:
                return Action.TOKEN_SHIFT, costVector
        if Action.TOKEN_SHIFT in bestLabels:
            return Action.TOKEN_SHIFT, costVector
        bestLabel = bestLabels.pop()
        return bestLabel, costVector

    def updateWithAction(self, action, new_state, action_probs, expert_action,
                         expert_action_taken, structuredInstance):
        self.actionsTaken.append(action)
        if action.label == Action.TOKEN_SHIFT:
            self.agenda.popleft()
        elif action.label == Action.TOKEN_EOS:
            self.agenda = deque([])
        else:
            self.tokensProduced.append(action)
        self.RNNState.append(new_state)
        self.actionProbsCache.append(action_probs)
        self.expertActions.append(expert_action)
        self.expertActionsTaken.append(expert_action_taken)


    '''
     Replace the action at indexed cell of the NLGState, and shorten the sequence up to and including the index.
     Initially, this method is used to replace the action at timestep of a roll-in sequence with an alternative action.
     Afterwards, it shortens the sequence so that the rest of it (after index) be recalculated by performing roll-out.
    '''
    def modifyAndShortenSequence(self, index, modification):
        modification.copyAttrValueTracking(self.actionsTaken[index])

        self.actionsTaken.set(index, modification)
        self.actionsTaken = self.actionsTaken[:index + 1]

    '''
     Returns a string representation of the word actions in the NLGState.
    '''
    def getWordSequenceToString(self):
        w = str()
        for action in self.actionsTaken:
            if action.label != Action.TOKEN_SHIFT and action.label != Action.TOKEN_EOS:
                w += action.label + " "
        return w.strip()

    '''
     Returns a string representation of the word actions in the NLGState, while omitting all punctuation.
    '''
    def getWordSequenceToString_NoPunct(self):
        w = str()
        for action in self.actionsTaken:
            if action.label != Action.TOKEN_SHIFT and action.label != Action.TOKEN_EOS and action.label != Action.TOKEN_PUNCT:
                w += action.label + " "
        return w.strip()

    '''
     Returns a string representation of the content actions in the NLGState.
    '''
    def getAttrSequenceToString(self):
        w = str()
        for action in self.actionsTaken:
            w += action.attribute + " "
        return w.strip()

    '''
     Returns the length of the sequence when not accounting for Action.TOKEN_SHIFT and Action.TOKEN_EOS actions.
    '''
    def getLength_NoBorderTokens(self):
        length = 0
        for action in self.actionsTaken:
            if action.label != Action.TOKEN_SHIFT and action.label != Action.TOKEN_EOS:
                length += 1
        return length

    '''
     Returns the length of the sequence when not accounting for Action.TOKEN_SHIFT and Action.TOKEN_EOS actions,
     and punctuation.
    '''
    def getLength_NoBorderTokens_NoPunct(self):
        length = 0
        for action in self.actionsTaken:
            if action.label != Action.TOKEN_SHIFT and action.label != Action.TOKEN_EOS and action.label != Action.TOKEN_PUNCT:
                length += 1
        return length

    '''
     Returns a subsequence consisting only of the content actions in the NLGState.
    '''
    def getAttributeSequence(self):
        attrSeq = []
        for action in self.actionsTaken:
            attrSeq.append(action.attribute)
        return attrSeq

    '''
     Returns a subsequence consisting only of the content actions in the NLGState, up to a specified index.
    '''
    def getAttributeSubSequence(self, index):
        attrSeq = []
        for action in self.actionsTaken[:index]:
            attrSeq.append(action.attribute)
        return attrSeq

    def __str__(self):
        return "NLGState{" + "sequence=" + self.actionsTaken.__str__() + '}'

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.actionsTaken.__str__() == other.actionsTaken.__str__()
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __gt__(self, other):
        return len(self.actionsTaken) > len(other.actionsTaken)

    def __ge__(self, other):
        return (len(self.actionsTaken) > len(other.actionsTaken)) or (len(self.actionsTaken) == len(other.actionsTaken))

    def __lt__(self, other):
        return len(self.actionsTaken) < len(other.actionsTaken)

    def __le__(self, other):
        return (len(self.actionsTaken) < len(other.actionsTaken)) or (len(self.actionsTaken) == len(other.actionsTaken))

    def __hash__(self):
        return hash(self.getWordSequenceToString())
