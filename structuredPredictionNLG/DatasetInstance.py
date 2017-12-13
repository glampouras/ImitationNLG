# Gerasimos Lampouras, 2017:
from Action import Action
from copy import copy
import imitation
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

'''
 This represents a single instance in the dataset. 
'''
class DatasetInstance(imitation.StructuredInstance):

    def __init__(self, MR, directReferenceSequence, directReference):
        self.input = MR
        self.output = NLGOutput()

        # A reference for the word actions of the DatasetInstance; this is constructed using the reference directly
        # corresponding to this instance in the dataset
        self.setDirectReferenceSequence(directReferenceSequence)
        # Realized string of the word actions in the direct reference
        self.directReference = directReference

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
                    if act.attribute != Action.TOKEN_EOS and act.attribute != Action.TOKEN_PUNCT and act.attribute != '[]':
                        self.directAttrSequence.append(Action(Action.TOKEN_SHIFT, act.attribute))
                    elif act.attribute == '[]' and act.word.startswith(Action.TOKEN_X):
                        self.directAttrSequence.append(Action(Action.TOKEN_SHIFT, act.word[3:act.word.find('_')]))
                    elif act.attribute == Action.TOKEN_EOS:
                        self.directAttrSequence.append(Action(Action.TOKEN_EOS, act.attribute))
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
                        if act.attribute != Action.TOKEN_EOS:
                            evaluationAttrSequence.append(Action(Action.TOKEN_SHIFT, act.attribute))
                        else:
                            evaluationAttrSequence.append(Action(Action.TOKEN_EOS, act.attribute))
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
                if act.attribute != Action.TOKEN_EOS:
                    self.directAttrSequence.append(Action(Action.TOKEN_SHIFT, act.attribute))
                else:
                    self.directAttrSequence.append(Action(Action.TOKEN_EOS, act.attribute))
            if act.attribute != Action.TOKEN_PUNCT:
                previousAttr = act.attribute


def cleanAndGetAttr(s):
    if "¬" in s:
        s = s[:s.find("¬")]
    if "=" in s:
        s = s[:s.find("=")]
    return s


def cleanAndGetValue(s):
    if "=" in s:
        return s[s.find("=") + 1:]
    return ""

class NLGOutput(imitation.StructuredOutput):
    def __init__(self):
        # References to be used during evaluation of this DatasetInstance
        self.evaluationReferences = set()
        self.evaluationReferenceSequences = []
        self.evaluationReferenceActionSequences = []
        self.evaluationReferenceActionSequences_that_follow_agenda = []
        self.chencherry = SmoothingFunction()

    # it must return an evalStats object with a loss
    def compareAgainst(self, predicted):
        evalStats = NLGEvalStats()

        maxBLEU = 0.0
        if predicted:
            weights = (0.25, 0.25, 0.25, 0.25)
            if len(predicted) < 4:
                weights = (1 / len(predicted),) * len(predicted)
            for ref in self.evaluationReferenceSequences:
                bleuOriginal = sentence_bleu([ref], predicted, weights, smoothing_function=self.chencherry.method2)
                if bleuOriginal > maxBLEU:
                    maxBLEU = bleuOriginal

                # todo resolve issues with Rouge library, add it to cost metric
                '''
                maxROUGE = 0.0;
                for ref in refs:
                    scores = rouge.get_scores(ref.lower(), gen.lower())
                    print(scores)
                    exit()
                    if bleuOriginal > maxROUGE:
                        maxROUGE = bleuOriginal
                return (maxBLEU + maxROUGE) / 2.0
                '''
        evalStats.BLEU = maxBLEU
        evalStats.loss = 1.0 - maxBLEU
        return evalStats


    # it must return an evalStats object with a loss
    def evaluate(self, predicted):
        evalStats = NLGEvalStats()

        maxBLEU = 0.0
        if predicted:
            if len(predicted) >= 4:
                for ref in self.evaluationReferences:
                    bleuOriginal = sentence_bleu([ref.split(" ")], predicted)
                    if bleuOriginal > maxBLEU:
                        maxBLEU = bleuOriginal

                    # todo resolve issues with Rouge library, add it to cost metric
                    '''
                    maxROUGE = 0.0;
                    for ref in refs:
                        scores = rouge.get_scores(ref.lower(), gen.lower())
                        print(scores)
                        exit()
                        if bleuOriginal > maxROUGE:
                            maxROUGE = bleuOriginal
                    return (maxBLEU + maxROUGE) / 2.0
                    '''
        evalStats.BLEU = maxBLEU
        return evalStats


# Then the NER eval stats
class NLGEvalStats(imitation.EvalStats):
    def __init__(self):
        super().__init__()
        self.BLEU = 0
        self.ROUGE = 0
        self.coverage = 0