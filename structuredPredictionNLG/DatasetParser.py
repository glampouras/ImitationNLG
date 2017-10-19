# Gerasimos Lampouras, 2017:
from Action import Action
from MeaningRepresentation import MeaningRepresentation
from DatasetInstance import DatasetInstance
from SimpleContentPredictor import SimpleContentPredictor
from NLGState import NLGState
import imitation
import os.path
import re
import Levenshtein
import _pickle as pickle
from nltk.util import ngrams

'''
 This is a general specification of a DatasetParser.
 A descendant of this class will need to be creater for every specific dataset
 (to deal with dataset-specific formats)
'''
class DatasetParser(object):

    def __init__(self, trainingFile, developmentFile, testingFile, datasetID, reset = False):
        self.singlePredicate = 'inform'
        self.dataset = datasetID

        if (reset or not self.loadTrainingLists()) and trainingFile:
            self.predicates = []
            self.attributes = {}
            self.valueAlignments = {}
            self.trainingInstances = {}
            self.vocabulary = set()

            self.attributes[self.singlePredicate] = set()
            self.trainingInstances[self.singlePredicate] = []

            self.maxWordSequenceLength = 0

            self.createLists(trainingFile, self.trainingInstances, True)

            # Create the evaluation refs for train data, as described in https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs
            for predicate in self.trainingInstances:
                for di in self.trainingInstances[predicate]:
                    refs = set()
                    refs.add(di.directReference)
                    refSeqs = [[o.label.lower() for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS]]
                    refActionSeqs = [[o for o in di.directReferenceSequence if o.label != Action.TOKEN_SHIFT]]
                    for di2 in self.trainingInstances[predicate]:
                        if di != di2 and di2.input.MRstr == di.input.MRstr:
                            refs.add(di2.directReference)
                            if di2.directReferenceSequence not in refSeqs:
                                refSeqs.append(o.label.lower() for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS)
                                refActionSeqs.append([o for o in di2.directReferenceSequence if o.label != Action.TOKEN_SHIFT])
                    di.output.evaluationReferences = refs
                    di.output.evaluationReferenceSequences = refSeqs
                    di.output.evaluationReferenceActionSequences = refActionSeqs
            self.writeTrainingLists()
        if (reset or not self.loadDevelopmentLists()) and developmentFile:
            self.developmentInstances = {}
            self.developmentInstances[self.singlePredicate] = []

            self.createLists(developmentFile, self.developmentInstances, False)

            # Create the evaluation refs for DEV data, as described in https://github.com/tuetschek/e2e-metrics/tree/master/example-inputs
            for predicate in self.developmentInstances:
                for di in self.developmentInstances[predicate]:
                    refs = set(di.directReference)
                    for di2 in self.developmentInstances[predicate]:
                        if di != di2 and di2.input.MRstr == di.input.MRstr:
                            refs.add(di2.directReference)
                    di.output.evaluationReferences = refs
            self.writeDevelopmentLists()
        if (reset or not self.loadTestingLists()) and testingFile:
            self.testingInstances = {}
            self.testingInstances[self.singlePredicate] = []

            self.createLists(testingFile, self.testingInstances, False)
            self.writeTestingLists()

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

            if refPart.startswith("\"") and refPart.endswith("\""):
                refPart = refPart[1:-1]

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
                    attributeValues[attribute] = value

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
                if forTrain:
                    self.vocabulary.add(word)

            alingedAttributes = set()
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
                            valuesToCompare = set()
                            valuesToCompare.update([value, attr])
                            valuesToCompare.update(value.split(" "))
                            valuesToCompare.update(attr.split(" "))
                            valuesToCompare.update(attr.split("_"))
                            for valueToCompare in valuesToCompare:
                                # obtain n-grams from the sentence
                                for n in range(1, 6):
                                    grams = ngrams(directReferenceSequence, n)

                                    # calculate the similarities between each gram and valueToCompare
                                    for gram in grams:
                                        if Action.TOKEN_X not in [o.label for o in gram].__str__() and Action.TOKEN_PUNCT not in [o.attribute for o in gram]:
                                            compare = " ".join(o.label for o in gram)
                                            backwardCompare = " ".join(o.label for o in reversed(gram))

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
                    bestGrams = {}

                    toRemove = set()
                    for value in observedValueAlignments.keys():
                        if observedValueAlignments[value]:
                            for gram, distance in observedValueAlignments[value]:
                                if distance > max:
                                    max = distance
                                    bestGrams = {}
                                if distance == max:
                                    bestGrams[gram] = value
                        else:
                            toRemove.add(value)
                    for value in toRemove:
                        del observedValueAlignments[value]

                    # Going with the earliest occurance of a matched ngram works best when aligning with hard alignments
                    # Because all the other match ngrams that occur to the left of the earliest, will probably be aligned as well
                    minOccurance = len(directReferenceSequence)
                    bestGram = False
                    bestValue = False
                    for gram in bestGrams:
                        occur = self.find_subList_in_actionList(gram, directReferenceSequence)[0]
                        if occur < minOccurance:
                            minOccurance = occur
                            bestGram = gram
                            bestValue = bestGrams[gram]

                    # Otherwise might be better to go for the longest ngram
                    '''
                    maxLen = 0
                    bestGram = False
                    bestValue = False
                    for gram in bestGrams:
                        if len(gram) > maxLen:
                            maxLen = distance
                            bestGram = gram
                            bestValue = bestGrams[gram]
                    '''

                    if bestGram:
                        # Find the subphrase that corresponds to the best aligned nGram
                        bestGramPos = self.find_subList_in_actionList(bestGram, directReferenceSequence)
                        if bestGramPos:
                            for i in range(bestGramPos[0], bestGramPos[1] + 1):
                                directReferenceSequence[i].attribute = valueToAttr[bestValue]
                                alingedAttributes.add(directReferenceSequence[i].attribute)
                            if forTrain:
                                # Store the best aligned nGram
                                if bestValue not in self.valueAlignments.keys():
                                    self.valueAlignments[bestValue] = {}
                                self.valueAlignments[bestValue][bestGram] = max
                            # And remove it from the observed ones for this instance
                            del observedValueAlignments[bestValue]
                        else:
                            observedValueAlignments[bestValue].remove((bestGram, max))
                for action in directReferenceSequence:
                    if action.label.startswith(Action.TOKEN_X):
                        action.attribute = action.label[3:action.label.find('_')]
                        alingedAttributes.add(action.attribute)


            # If not all attributes are aligned, ignore the instance from training?
            # Alternatively, we could align them randomly; certainly not ideal, but usually it concerns edge cases
            if MR.attributeValues.keys() == alingedAttributes or not forTrain:
                directReferenceSequence = inferNaiveAlignments(directReferenceSequence)
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
            if nlWord.label != Action.TOKEN_EOS and nlWord.label != Action.TOKEN_SHIFT and nlWord.label != Action.TOKEN_PUNCT:
                if nlWord.label.startswith(Action.TOKEN_X):
                    cleanedWords += " " + mr.delexicalizationMap[nlWord.label]
                else:
                    cleanedWords += " " + nlWord.label
        cleanedWords = cleanedWords.strip()
        if not cleanedWords.endswith("."):
            cleanedWords += " ."
        return cleanedWords.strip()

    @staticmethod
    def find_subList_in_actionList(sl, l):
        sll = len(sl)
        for ind in (i for i, e in enumerate(l) if e.label == sl[0].label):
            if [o.label for o in l[ind:ind + sll]] == [r.label for r in sl]:
                return ind, ind + sll - 1


def inferNaiveAlignments(sequence, useHardAlignments=True):
    attrSeq = [o.attribute for o in sequence]
    if useHardAlignments:
        currentAttr = ""
        for act in sequence:
            if (act.attribute == "[]" or act.attribute == Action.TOKEN_PUNCT) and currentAttr != "":
                act.attribute = currentAttr
            currentAttr = act.attribute

        currentAttr = ""
        for act in reversed(sequence):
            if (act.attribute == "[]" or act.attribute == Action.TOKEN_PUNCT) and currentAttr != "":
                act.attribute = currentAttr
            currentAttr = act.attribute
    else:
        while True:
            changes = {}
            for i, attr in enumerate(attrSeq):
                if attr != "[]" and attr != Action.TOKEN_PUNCT:
                    if i - 1 >= 0 and attrSeq[i - 1] == "[]":
                        if attr not in changes:
                            changes[attr] = set()
                        changes[attr].add(i - 1)
                    if i + 1 < len(attrSeq) and attrSeq[i + 1] == "[]":
                        if attr not in changes:
                            changes[attr] = set()
                        changes[attr].add(i + 1)
            for attr in changes:
                for index in changes[attr]:
                    attrSeq[index] = attr
                    sequence[index].attribute = attr
            if not changes:
                break
        while "[]" in attrSeq:
            index = attrSeq.index("[]")
            copyFrom = index - 1
            while copyFrom >= 0:
                if attrSeq[copyFrom] != "[]" and attrSeq[copyFrom] != Action.TOKEN_PUNCT:
                    attrSeq[index] = attrSeq[copyFrom]
                    sequence[index].attribute = attrSeq[copyFrom]
                    copyFrom = -1
                else:
                    copyFrom -= 1
            if attrSeq[index] == "[]":
                copyFrom = index + 1
                while copyFrom < len(attrSeq):
                    if attrSeq[copyFrom] != "[]" and attrSeq[copyFrom] != Action.TOKEN_PUNCT:
                        attrSeq[index] = attrSeq[copyFrom]
                        sequence[index].attribute = attrSeq[copyFrom]
                        copyFrom = len(attrSeq)
                    else:
                        copyFrom += 1
        while Action.TOKEN_PUNCT in attrSeq:
            index = attrSeq.index(Action.TOKEN_PUNCT)
            if index > 0:
                attrSeq[index] = attrSeq[index - 1]
                sequence[index].attribute = attrSeq[index - 1]
            else:
                attrSeq[index] = attrSeq[index + 1]
                sequence[index].attribute = attrSeq[index + 1]
    currentAttr = ""
    for i, act in enumerate(sequence):
        if act.attribute != currentAttr:
            sequence.insert(i, Action(Action.TOKEN_SHIFT, act.attribute))
            currentAttr = act.attribute
    sequence.append(Action(Action.TOKEN_EOS, Action.TOKEN_EOS))
    return sequence


if __name__ == '__main__':
    # load the training data!
    # parser = DatasetParser(r'../data/trainset.csv', r'../data/devset.csv', r'../data/test_e2e.csv', 'E2E', True)
    parser = DatasetParser(r'../toyData/toy_trainset.csv', r'../toyData/toy_devset.csv', False, 'toy_E2E', True)

    # Test that loading was correct
    '''
    print(parser.vocabulary)
    print(len(parser.trainingInstances))
    for di in parser.trainingInstances[parser.singlePredicate]:
        print(di.input.MRstr)
        print(di.input.attributeValues)
        print(di.input.delexicalizationMap)
        print(di.getDirectReferenceAttrValueSequence())
        print(di.directReferenceSequence)
        print()
    '''

    # Example of training and using the SimpleContentPredictor
    '''
    # avgBLEU = 0.0
    contentPredictor = SimpleContentPredictor(parser.dataset, parser.attributes, parser.trainingInstances)
    for di in parser.developmentInstances[parser.singlePredicate]:
        print(di.input.MRstr)
        print(di.input.attributeValues)
        print(di.directReference)
        refCont = [o.attribute for o in di.getDirectReferenceAttrValueSequence()]
        genCont = contentPredictor.rollContentSequence_withLearnedPolicy(di)
        # BLEU = sentence_bleu([refCont], genCont)
        print(refCont)
        print(genCont)
        # print(BLEU)
        print()

        # avgBLEU += BLEU
    # avgBLEU /= len(self.developmentInstances[self.singlePredicate])
    # print('==========')
    # print(avgBLEU)
    '''

    # Example of using the expert policy for word prediction
    contentPredictor = SimpleContentPredictor(parser.dataset, parser.attributes, parser.trainingInstances)
    initialState = NLGState(contentPredictor, parser.trainingInstances[parser.singlePredicate][0])

    learner = imitation.ImitationLearner()
    output = learner.predict(parser.trainingInstances[parser.singlePredicate][0], initialState, 1.0)
    print(output)

