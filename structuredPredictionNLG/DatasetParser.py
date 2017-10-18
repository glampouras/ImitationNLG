# Gerasimos Lampouras, 2017:
from Action import Action
from MeaningRepresentation import MeaningRepresentation
from DatasetInstance import DatasetInstance
from SimpleContentPredictor import SimpleContentPredictor
from WordPredictor import getExpertPolicyWordAction
import os.path
import re
import Levenshtein
import _pickle as pickle
from nltk.util import ngrams
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

'''
 This is a general specification of a DatasetParser.
 A descendant of this class will need to be creater for every specific dataset
 (to deal with dataset-specific formats)
'''
class DatasetParser(object):

    def __init__(self, trainingFile, developmentFile, testingFile, datasetID):
        self.singlePredicate = 'inform'
        self.dataset = datasetID

        reset = True

        if (reset or not self.loadTrainingLists()) and trainingFile:
            self.predicates = []
            self.attributes = {}
            self.valueAlignments = {}
            self.trainingInstances = {}

            self.attributes[self.singlePredicate] = set()
            self.trainingInstances[self.singlePredicate] = []

            self.maxWordSequenceLength = 0

            self.createLists(trainingFile, self.trainingInstances, True)

            # Create the evaluation refs for train data, as described in https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs
            for predicate in self.trainingInstances:
                for di in self.trainingInstances[predicate]:
                    refs = set()
                    refSeqs = set()
                    for di2 in self.trainingInstances[predicate]:
                        if di != di2 and di2.MR.MRstr == di.MR.MRstr:
                            refs.add(di2.directReference)
                            refSeqs.add(di2.directReferenceSequence)
                    di.evaluationReferences = refs
                    di.evaluationReferenceSequences = refSeqs
            self.writeTrainingLists()
        if (reset or not self.loadDevelopmentLists()) and developmentFile:
            self.developmentInstances = {}
            self.developmentInstances[self.singlePredicate] = []

            self.createLists(developmentFile, self.developmentInstances, False)

            # Create the evaluation refs for DEV data, as described in https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs
            for predicate in self.developmentInstances:
                for di in self.developmentInstances[predicate]:
                    refs = set()
                    for di2 in self.developmentInstances[predicate]:
                        if di != di2 and di2.MR.MRstr == di.MR.MRstr:
                            refs.add(di2.directReference)
                    di.evaluationReferences = refs
            self.writeDevelopmentLists()
        if (reset or not self.loadTestingLists()) and testingFile:
            self.testingInstances = {}
            self.testingInstances[self.singlePredicate] = []

            self.createLists(testingFile, self.testingInstances, False)
            self.writeTestingLists()

        # Test that loading was correct
        '''
        print(len(self.trainingInstances))
        for di in self.trainingInstances[self.singlePredicate]:
            print(di.MR.MRstr)
            print(di.MR.attributeValues)
            print(di.MR.delexicalizationMap)
            print(di.getDirectReferenceAttrValueSequence())
            print(di.directReferenceSequence)
            print()
        '''

        # Example of training and using the SimpleContentPredictor
        '''
        avgBLEU = 0.0
        self.contentPredictor = SimpleContentPredictor(self.dataset, self.attributes, self.trainingInstances)
        for di in self.developmentInstances[self.singlePredicate]:
            print(di.MR.MRstr)
            print(di.MR.attributeValues)
            refCont = [o.attribute for o in di.getDirectReferenceAttrValueSequence()]
            genCont = self.contentPredictor.rollContentSequence_withLearnedPolicy(di)
            BLEU = sentence_bleu([refCont], genCont)
            print(refCont)
            print(genCont)
            print(BLEU)
            print()

            avgBLEU += BLEU
        avgBLEU /= len(self.developmentInstances[self.singlePredicate])
        print('==========')
        print(avgBLEU)
        '''

        # Example of using the expert policy for word prediction
        di = self.developmentInstances[self.singlePredicate][0]
        sequence = []
        print(getExpertPolicyWordAction(sequence, di))

    def createLists(self, dataFile, instances, forTrain = False):
        print("Create lists from ", dataFile, "...")

        dataPart = []

        # We read the data from the data files.
        with open(dataFile, encoding="utf8") as f:
            lines = f.readlines()
            for s in lines:
                s = str(s)
                if s.startswith("\""):
                    dataPart.append(s)

        # This dataset has no predicates, so we assume a default predicate
        self.predicates.append(self.singlePredicate)
        num = 0
        # Each line corresponds to a MR
        for line in dataPart:
            num += 1

            if "\"," in line:
                MRPart = line.split("\",")[0]
                refPart = line.split("\",")[1].lower()
            else:
                MRPart = line
                refPart = ""

            if MRPart.startswith("\""):
                MRPart = MRPart[1:]
            if refPart.startswith("\""):
                refPart = refPart[1:]
            if refPart.endswith("\""):
                refPart = refPart[:-1]
            refPart = re.sub("([.,?:;!'-])", " \g<1> ", refPart)
            refPart = refPart.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace("  ", " ").strip()
            MRAttrValues = MRPart.split(",")

            # Map from original values to delexicalized values
            delexicalizedMap = {}
            # Map attributes to their values
            attributeValues = {}

            for attrValue in MRAttrValues:
                value = attrValue[attrValue.find("[") + 1:attrValue.find("]")].strip().lower()
                attribute = attrValue[0:attrValue.find("[")].strip().lower().replace(" ", "_")

                if "name" in attribute or "near" in attribute:
                    delexValue = Action.TOKEN_X + attribute + "_0"
                    delexicalizedMap[delexValue] = value
                    value = delexValue
                if value == "yes" or value == "no":
                    value = attribute + "_" + value

                if forTrain and self.singlePredicate not in self.attributes:
                    self.attributes[self.singlePredicate] = set()
                if attribute:
                    if forTrain:
                        self.attributes[self.singlePredicate].add(attribute)
                    if attribute not in attributeValues:
                        attributeValues[attribute] = set()
                    if value:
                        attributeValues[attribute].add(value)

            for deValue in delexicalizedMap.keys():
                value = delexicalizedMap[deValue]
                if (" " + value.lower() + " ") in (" " + refPart + " "):
                    refPart = (" " + refPart + " ").replace((" " + value.lower() + " "), (" " + deValue + " ")).strip()
                elif value.lower() in refPart:
                    refPart = refPart.replace(value.lower(), deValue).strip()

            observedWordSequence = []

            words = refPart.replace(", ,", " , ").replace(". .", " . ").replace("  ", " ").split(" ")
            for word in words:
                if "0f" in word:
                    word = word.replace("0f", "of")

                m = re.search("^@x@([a-z]+)_([0-9]+)", word)
                if m and m.group(0) != word:
                    var = m.group(0);
                    realValue = delexicalizedMap.get(var)
                    realValue = word.replace(var, realValue)
                    delexicalizedMap[var] = realValue
                    observedWordSequence.append(var.strip())
                else:
                    m = re.match("([0-9]+)([a-z]+)", word)
                    if m and m.group(1).strip() == "o":
                        observedWordSequence.add(m.group(1).strip() + "0")
                    elif m:
                        observedWordSequence.append(m.group(1).strip())
                        observedWordSequence.append(m.group(2).strip())
                    else:
                        m = re.match("([a-z]+)([0-9]+)", word)
                        if m and (m.group(1).strip() == "l" or m.group(1).strip() == "e"):
                            observedWordSequence.append("£" + m.group(2).strip())
                        elif m:
                            observedWordSequence.append(m.group(1).strip())
                            observedWordSequence.append(m.group(2).strip())
                        else:
                            m = re.match("(£)([a-z]+)", word)
                            if m:
                                observedWordSequence.append(m.group(1).strip())
                                observedWordSequence.append(m.group(2).strip())
                            else:
                                m = re.match("([a-z]+)(£[0-9]+)", word)
                                if m:
                                    observedWordSequence.append(m.group(1).strip())
                                    observedWordSequence.append(m.group(2).strip())
                                else:
                                    m = re.match("([0-9]+)([a-z]+)([0-9]+)", word)
                                    if m:
                                        observedWordSequence.append(m.group(1).strip())
                                        observedWordSequence.append(m.group(2).strip())
                                        observedWordSequence.append(m.group(3).strip())
                                    else:
                                        m = re.match("([0-9]+)(@x@[a-z]+_0)", word)
                                        if m:
                                            observedWordSequence.append(m.group(1).strip())
                                            observedWordSequence.append(m.group(2).strip())
                                        else:
                                            m = re.match("(£[0-9]+)([a-z]+)", word)
                                            if m and m.group(2).strip() == "o":
                                                observedWordSequence.append(m.group(1).strip() + "0")
                                            else:
                                                observedWordSequence.append(word.strip())

            MR = MeaningRepresentation(self.singlePredicate, attributeValues, MRPart, delexicalizedMap)

            # We store the maximum observed word sequence length, to use as a limit during generation
            if forTrain and len(observedWordSequence) > self.maxWordSequenceLength:
                self.maxWordSequenceLength = len(observedWordSequence)

            # We initialize the alignments between words and attribute/value pairs
            wordToAttrValueAlignment = []
            for word in observedWordSequence:
                if re.match("[.,?:;!'\"]", word.strip()):
                    wordToAttrValueAlignment.append(Action.TOKEN_PUNCT)
                else:
                    wordToAttrValueAlignment.append("[]")
            directReferenceSequence = []
            for r, word in enumerate(observedWordSequence):
                directReferenceSequence.append(Action(word, wordToAttrValueAlignment[r]))

            if directReferenceSequence:
                # Align subphrases of the sentence to attribute values
                observedValueAlignments = {}
                valueToAttr = {}
                for attr in MR.attributeValues.keys():
                    values = sorted(list(MR.attributeValues[attr]))
                    for value in values:
                        if not value.startswith(Action.TOKEN_X):
                            observedValueAlignments[value] = set()
                            valueToAttr[value] = attr
                            valuesToCompare = [value, attr]
                            for valueToCompare in valuesToCompare:
                                # obtain n-grams from the sentence
                                for n in range(1, 6):
                                    grams = ngrams(directReferenceSequence, n)

                                    # calculate the similarities between each gram and valueToCompare
                                    for gram in grams:
                                        if Action.TOKEN_X not in gram and Action.TOKEN_PUNCT not in gram:
                                            compare = " ".join(o.word for o in gram)
                                            backwardCompare = " ".join(o.word for o in reversed(gram))

                                            if compare.strip():
                                                # Calculate the character-level distance between the value and the nGram (in its original and reversed order)
                                                distance = Levenshtein.ratio(valueToCompare.lower(), compare.lower())
                                                backwardDistance = Levenshtein.ratio(valueToCompare.lower(), backwardCompare.lower())

                                                # We keep the best distance score; note that the Levenshtein distance is normalized so that greater is better
                                                if backwardDistance > distance:
                                                    distance = backwardDistance
                                                if (distance > 0.3):
                                                    observedValueAlignments[value].add((gram, distance))

                while observedValueAlignments.keys():
                    # Find the best aligned nGram
                    max = -1000
                    bestValue = False
                    bestGram = False

                    toRemove = set()
                    for value in observedValueAlignments.keys():
                        if observedValueAlignments[value]:
                            for gram, distance in observedValueAlignments[value]:
                                if distance > max:
                                    max = distance
                                    bestValue = value
                                    bestGram = gram
                        else:
                            toRemove.add(value)
                    for value in toRemove:
                        del observedValueAlignments[value]

                    if bestGram:
                        # Find the subphrase that corresponds to the best aligned nGram
                        bestGramPos = self.find_subList_in_actionList(bestGram, directReferenceSequence)
                        if bestGramPos:
                            for i in range(bestGramPos[0], bestGramPos[1] + 1):
                                directReferenceSequence[i].attribute = valueToAttr[bestValue]
                            if forTrain:
                                # Store the best aligned nGram
                                if bestValue not in self.valueAlignments.keys():
                                    self.valueAlignments[bestValue] = {}
                                self.valueAlignments[bestValue][bestGram] = max
                            # And remove it from the observed ones for this instance
                            del observedValueAlignments[bestValue]
                        else:
                            observedValueAlignments[value].remove((bestGram, max))
                for action in directReferenceSequence:
                    if action.word.startswith(Action.TOKEN_X):
                        action.attribute = action.word[3:action.word.find('_')]
            DI = DatasetInstance(MR, directReferenceSequence, self.postProcessRef(MR, directReferenceSequence))
            instances[self.singlePredicate].append(DI)

    def loadTrainingLists(self):
        print("Attempting to load training data...")
        self.predicates = False
        self.attributes = False
        self.valueAlignments = False
        self.trainingInstances = False
        self.maxWordSequenceLength = False

        if os.path.isfile('../cache/predicates_' + self.dataset + '.pickle'):
            with open('../cache/predicates_' + self.dataset + '.pickle', 'rb') as handle:
                self.predicates = pickle.load(handle)
        if os.path.isfile('../cache/attributes_' + self.dataset + '.pickle'):
            with open('../cache/attributes_' + self.dataset + '.pickle', 'rb') as handle:
                self.attributes = pickle.load(handle)
        if os.path.isfile('../cache/valueAlignments_' + self.dataset + '.pickle'):
            with open('../cache/valueAlignments_' + self.dataset + '.pickle', 'rb') as handle:
                self.valueAlignments = pickle.load(handle)
        if os.path.isfile('../cache/trainingInstances_' + self.dataset + '.pickle'):
            with open('../cache/trainingInstances_' + self.dataset + '.pickle', 'rb') as handle:
                self.trainingInstances = pickle.load(handle)
        if os.path.isfile('../cache/maxWordSequenceLength_' + self.dataset + '.pickle'):
            with open('../cache/maxWordSequenceLength_' + self.dataset + '.pickle', 'rb') as handle:
                self.maxWordSequenceLength = pickle.load(handle)

        if self.predicates and self.attributes and self.valueAlignments and self.trainingInstances and self.maxWordSequenceLength:
            return True
        return False

    def loadDevelopmentLists(self):
        print("Attempting to load development data...")
        self.developmentInstances = False

        if os.path.isfile('../cache/developmentInstances_' + self.dataset + '.pickle'):
            with open('../cache/developmentInstances_' + self.dataset + '.pickle', 'rb') as handle:
                self.developmentInstances = pickle.load(handle)

        if self.developmentInstances:
            return True
        return False

    def loadTestingLists(self):
        print("Attempting to load testing data...")
        self.testingInstances = False

        if os.path.isfile('../cache/testingInstances_' + self.dataset + '.pickle'):
            with open('../cache/testingInstances_' + self.dataset + '.pickle', 'rb') as handle:
                self.testingInstances = pickle.load(handle)

        if self.testingInstances:
            return True
        return False
            
    def writeTrainingLists(self):
        print("Writing training data...")
        with open('../cache/predicates_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.predicates, handle)
        with open('../cache/attributes_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.attributes, handle)
        with open('../cache/valueAlignments_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.valueAlignments, handle)
        with open('../cache/trainingInstances_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.trainingInstances, handle)
        with open('../cache/maxWordSequenceLength_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.maxWordSequenceLength, handle)

    def writeDevelopmentLists(self):
        print("Writing development data...")
        with open('../cache/developmentInstances_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.developmentInstances, handle)

    def writeTestingLists(self):
        print("Writing testing data...")
        with open('../cache/testingInstances_' + self.dataset + '.pickle', 'wb') as handle:
            pickle.dump(self.testingInstances, handle)

    @staticmethod
    def postProcessRef(mr, refSeq):
        cleanedWords = ""
        for nlWord in refSeq:
            if nlWord.word != Action.TOKEN_END and nlWord.word != Action.TOKEN_START and nlWord.word != Action.TOKEN_PUNCT:
                if nlWord.word.startswith(Action.TOKEN_X):
                    cleanedWords += " " + mr.delexicalizationMap[nlWord.word]
                else:
                    cleanedWords += " " + nlWord.word
        cleanedWords = cleanedWords.strip()
        if not cleanedWords.endswith("."):
            cleanedWords += " ."
        return cleanedWords.strip()

    @staticmethod
    def find_subList_in_actionList(sl, l):
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e.word == sl[0].word):
            if [o.word for o in l[ind:ind + sll]] == [r.word for r in sl]:
                return ind, ind + sll - 1


if __name__ == '__main__':
    parser = DatasetParser(r'../data/trainset.csv', r'../data/devset.csv', r'../data/test_e2e.csv', 'E2E')
