# Gerasimos Lampouras, 2017:
import nltk
import itertools
import os.path
import _pickle as pickle
from collections import Counter

class SimpleContentPredictor(object):

    def __init__(self, datasetID, attributes, trainingInstances):
        self.dataset = datasetID
        self.trainingLen = {}
        for predicate in attributes:
            self.trainingLen[predicate] = len(trainingInstances[predicate])

        if not self.loadContentPredictor():
            for predicate in attributes:
                self.unigram_counts = {}
                self.bigram_counts = {}
                self.trigram_counts = {}
                self.unigram_counts[predicate] = Counter(attributes[predicate])
                bigrams = []
                trigrams = []
                for di in trainingInstances[predicate]:
                    seq = [o.attribute for o in di.getDirectReferenceAttrValueSequence()]
                    bigrams.extend(nltk.bigrams(seq, pad_left=True, pad_right=True))
                    trigrams.extend(nltk.trigrams(seq, pad_left=True, pad_right=True))
                self.bigram_counts[predicate] = Counter(bigrams)
                self.trigram_counts[predicate] = Counter(trigrams)
            self.writeContentPredictor()

    def getLMProbability(self, predicate, sentence_x, smoothing=0.0):
        unique_words = len(self.unigram_counts.keys()) + 2 # For the None paddings
        x_bigrams = nltk.bigrams(sentence_x, pad_left=True, pad_right=True)
        x_trigrams = nltk.trigrams(sentence_x, pad_left=True, pad_right=True)
        prob_x = 1.0
        for bg in x_bigrams:
            if bg[0] == None:
                prob_bg = (self.bigram_counts[predicate][bg] + smoothing) / (self.trainingLen[predicate] + smoothing * unique_words)
            else:
                prob_bg = (self.bigram_counts[predicate][bg] + smoothing) / (self.unigram_counts[predicate][bg[0]] + smoothing * unique_words)
            prob_x = prob_x * prob_bg
        for bg in x_trigrams:
            prob_bg = (self.trigram_counts[predicate][bg] + smoothing) / (self.bigram_counts[predicate][bg[:-1]] + smoothing * unique_words)
            prob_x = prob_x * prob_bg
        return prob_x

    def rollContentSequence_withLearnedPolicy(self, datasetInstance, contentSequence=False):
        if not contentSequence:
            contentSequence = []

        attrs = set(datasetInstance.MR.attributeValues.keys())
        for attr in contentSequence:
            attrs.remove(attr)
        permutations = itertools.permutations(attrs)

        bestPermut = False
        max = -1
        for permut in permutations:
            prob = self.getLMProbability(datasetInstance.MR.predicate, list(permut), 1.0)
            if prob > max:
                max = prob
                bestPermut = permut

        seq = contentSequence[:]
        seq.extend(bestPermut)
        return seq

    '''
    # TODO: Implement/fix expert policy for content sequence (if we actually need it)
    def rollContentSequence_withExpertPolicy(self, datasetInstance, rollInSequence):
        minCost = 1.0
        for refSeq in datasetInstance.getEvaluationReferenceAttrValueSequences():
            currentAttr = rollInSequence.sequence[- 1].attribute

            rollOutList = rollInSequence.sequence[:]
            refList = refSeq.sequence[:]

            if len(rollOutList) < len(refList):
                if currentAttr == Action.TOKEN_END:
                    while len(rollOutList) == len(refList):
                        rollOutList.append(Action("££", "££"))
                else:
                    rollOutList.extend(refList.subList[len(rollInSequence.sequence()):])
            else:
                while len(rollOutList) != len(refList):
                    refList.append(Action("££", "££"))

            rollOut = ActionSequence(rollOutList).getAttrSequenceToString().lower().strip()
            newRefSeq = ActionSequence(refList)
            refWindows = []
            refWindows.append(newRefSeq.getAttrSequenceToString().lower().strip())

            totalAttrValuesInRef = 0;
            attrValuesInRefAndNotInRollIn = 0;
            for attrValueAct in refList:
                if attrValueAct.attribute != Action.TOKEN_END:
                    totalAttrValuesInRef += 1

                    containsAttrValue = False
                    for a in rollOutList:
                        if a.attribute == attrValueAct.attribute:
                            containsAttrValue = True;
                            break;
                    if not containsAttrValue:
                        attrValuesInRefAndNotInRollIn != 1
            coverage = attrValuesInRefAndNotInRollIn / totalAttrValuesInRef
            #System.out.println("ROLLOUT " + rollOut);
            #System.out.println("REFS " + refWindows);
            refCost = LossFunction.getCostMetric(rollOut, refWindows, coverage);
            if refCost < minCost:
                minCost = refCost;
        return minCost
    '''

    def writeContentPredictor(self):
        print("Writing content predictor...")

        with open('../cache/contentPredictor_unigram_counts_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.unigram_counts, handle)
        with open('../cache/contentPredictor_bigram_counts_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.bigram_counts, handle)
        with open('../cache/contentPredictor_trigram_counts_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.trigram_counts, handle)

    def loadContentPredictor(self):
        print("Attempting to load content predictor...")

        self.unigram_counts = False
        self.bigram_counts = False

        if os.path.isfile('../cache/contentPredictor_unigram_counts_' + self.dataset + '.pickle'):
            with open('../cache/contentPredictor_unigram_counts_' + self.dataset + '.pickle', 'rb') as handle:
                self.unigram_counts = pickle.load(handle)
        if os.path.isfile('../cache/contentPredictor_bigram_counts_' + self.dataset + '.pickle'):
            with open('../cache/contentPredictor_bigram_counts_' + self.dataset + '.pickle', 'rb') as handle:
                self.bigram_counts = pickle.load(handle)
        if os.path.isfile('../cache/contentPredictor_trigram_counts_' + self.dataset + '.pickle'):
            with open('../cache/contentPredictor_trigram_counts_' + self.dataset + '.pickle', 'rb') as handle:
                self.trigram_counts = pickle.load(handle)

        if self.unigram_counts and self.bigram_counts and self.trigram_counts:
            return True
        return False