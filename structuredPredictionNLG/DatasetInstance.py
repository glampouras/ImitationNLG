# Gerasimos Lampouras, 2017:
from Action import Action
from copy import copy

'''
 This is an abstract specification of a DatasetParser.
'''
class DatasetInstance(object):

    def __init__(self, MR, directReferenceSequence, directReference):
        # The meaning representation
        self.MR = MR

        # A reference for the word actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset
        self.setDirectReferenceSequence(directReferenceSequence)
        # Realized string of the word actions in the direct reference
        self.directReference = directReference

        # References to be used during evaluation of this DatasetInstance
        self.evaluationReferences = set()
        self.evaluationReferences.add(directReference)

        # A reference for the word actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset and not processed further
        # Primarily used as a cache for the reference, to be used in "cache baselines"
        self.originalDirectReferenceSequence = False
        # A reference for the content actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset
        self.directAttrSequence = False
        # File from which DI was parsed; may be used for training/development/testing seperation
        self.originFile = False
        # References for the content actions of the DatasetInstance; this is constructed using the references
        # corresponding to any identical instance in the dataset
        self.evaluationAttrSequences = False

    '''
     Clone constructor.
     @param other DatasetInstance whose values will be used to instantiate this object
    '''
    def copy(self, other):
        self.MR = copy.copy(other.MR)

        self.setDirectReferenceSequence(copy.copy(other.directReferenceSequence))
        self.directReference = copy.copy(other.directReference)

        self.evaluationReferences = set()
        self.evaluationReferences.add(copy.copy(other.directReference))

    '''
     Returns (and constructs when first called) a sequence of content actions
     based on the direct reference of this DatasetInstance.
     @return A sequence of content actions.
    '''
    def getDirectReferenceAttrValueSequence(self):
        if not self.directAttrSequence and self.directReferenceSequence:
            self.directAttrSequence = []
            previousAttr = ""
            for act in self.directReferenceSequence:
                if act.attribute != previousAttr:
                    if act.attribute != Action.TOKEN_END and act.attribute != Action.TOKEN_PUNCT and act.attribute != '[]':
                        self.directAttrSequence.append(Action(Action.TOKEN_START, act.attribute))
                    elif act.attribute == '[]' and act.word.startswith(Action.TOKEN_X):
                        self.directAttrSequence.append(Action(Action.TOKEN_START, act.word[3:act.word.find('_')]))
                    elif act.attribute == Action.TOKEN_END:
                        self.directAttrSequence.append(Action(Action.TOKEN_END, act.attribute))
                if act.attribute != Action.TOKEN_PUNCT:
                    previousAttr = act.attribute
        return self.directAttrSequence

    '''
     Returns (and constructs when first called) a sequence of content actions
     based on the direct reference of this DatasetInstance.
     @return A sequence of content actions.
    '''
    '''
    def getEvaluationReferenceAttrValueSequences(self):
        if not self.evaluationAttrSequences and self.evaluationReferenceSequences:
            self.evaluationAttrSequences = set()
            for evaluationReferenceSequence in self.evaluationReferenceSequences:
                evaluationAttrSequence = []
                previousAttr = ""
                for act in evaluationReferenceSequence:
                    if act.attribute != previousAttr:
                        if act.attribute != Action.TOKEN_END:
                            evaluationAttrSequence.append(Action(Action.TOKEN_START, act.attribute))
                        else:
                            evaluationAttrSequence.append(Action(Action.TOKEN_END, act.attribute))
                    if act.attribute != Action.TOKEN_PUNCT:
                        previousAttr = act.attribute
                self.evaluationAttrSequences.append(evaluationAttrSequence)
        return self.evaluationAttrSequences
    '''

    '''
     Sets the word action sequence (and also constructs the corresponding content action sequence) 
     to be used as direct reference sequence for the DatasetInstance.
     @param directReferenceSequence The word action sequence to be set.
    '''
    def setDirectReferenceSequence(self, directReferenceSequence):
        self.directReferenceSequence = directReferenceSequence
        self.directAttrSequence = []
        previousAttr = ""
        for act in directReferenceSequence:
            if act.attribute != previousAttr:
                if act.attribute != Action.TOKEN_END:
                    self.directAttrSequence.append(Action(Action.TOKEN_START, act.attribute))
                else:
                    self.directAttrSequence.append(Action(Action.TOKEN_END, act.attribute))
            if act.attribute != Action.TOKEN_PUNCT:
                previousAttr = act.attribute