# Gerasimos Lampouras, 2017:
from collections import defaultdict
from Action import Action
from MeaningRepresentation import MeaningRepresentation
from DatasetInstance import DatasetInstance
import re

'''
 This is a general specification of a DatasetParser.
 A descendant of this class will need to be creater for every specific dataset
 (to deal with dataset-specific formats)
'''
class DatasetParser(object):

    def __init__(self, dataFile):
        self.singlePredicate = 'inform'

        self.predicates = []
        self.attributes = defaultdict()
        self.attributeValuePairs = defaultdict()
        self.valueAlignments = defaultdict()
        self.datasetInstances = defaultdict()

        self.attributes[self.singlePredicate] = set()
        self.datasetInstances[self.singlePredicate] = []

        self.maxWordSequenceLength = 0
        self.maxContentSequenceLength = 0

        self.createLists(dataFile)
        # Test that loading was correct
        print(len(self.datasetInstances))
        for di in self.datasetInstances[self.singlePredicate]:
            print(di.MR.MRstr)
            print(di.MR.attributeValues)
            print(di.MR.delexicalizationMap)
            print(di.directReference)
            print()

    def createLists(self, dataFile):
        print("create lists")

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
        # This actually fixes nothing...
        # Fix some errors in the data set
        '''
        for i, s in enumerate(dataPart):
            if "\"," not in s:
                line = dataPart[i - 1]
                line += s
                dataPart[i - 1] = line
                dataPart.remove(i)
        '''

        num = 0
        # Each line corresponds to a MR
        for line in dataPart:
            num += 1

            MRPart = line.split("\",")[0]
            refPart = line.split("\",")[1].lower()

            if MRPart.startswith("\""):
                MRPart = MRPart[1:]
            if refPart.startswith("\""):
                refPart = refPart[1:]
            if refPart.endswith("\""):
                refPart = refPart[:-1]
            refPart = refPart.replace("\\?", " \\? ").replace("\\.", " \\.").replace(",", " , ").replace("  ", " ").strip()
            MRAttrValues = MRPart.split(",")

            # Map from original values to delexicalized values
            delexicalizedMap = defaultdict()
            # Map attributes to their values
            attributeValues = defaultdict()

            for attrValue in MRAttrValues:
                value = attrValue[attrValue.find("[") + 1:attrValue.find("]")].strip().lower()
                attribute = attrValue[0:attrValue.find("[")].strip().lower().replace(" ", "_")

                if "name" in attribute or "near" in attribute:
                    delexValue = Action.TOKEN_X + attribute + "_0"
                    delexicalizedMap[delexValue] = value
                    value = delexValue
                if value == "yes" or value == "no":
                    value = attribute + "_" + value
                self.attributes[self.singlePredicate] = set()
                if attribute:
                    self.attributes[self.singlePredicate].add(attribute)
                    if attribute not in self.attributeValuePairs:
                        self.attributeValuePairs[attribute] = set()
                    if attribute not in attributeValues:
                        attributeValues[attribute] = set()
                    if value:
                        self.attributeValuePairs[attribute].add(value)
                        attributeValues[attribute].add(value)

            for deValue in delexicalizedMap.keys():
                value = delexicalizedMap[deValue]
                if (" " + value.lower() + " ") in (" " + refPart + " "):
                    refPart = (" " + refPart + " ").replace((" " + value.lower() + " "), (" " + deValue + " ")).strip()

            observedWordSequence = []

            words = refPart.replace(", ,", " , ").replace(". .", " . ").replace("[.,?:;!'-]", " $0 ").replace("  ", " ").split(" ")
            for word in words:
                if "0f" in word:
                    word = word.replace("0f", "of")

                '''
                TO-DO: Fix these python patterns
                p0 = re.compile("^@x@([a-z]+)_([0-9]+)")
                p1 = re.compile("([0-9]+)([a-z]+)")
                p2 = re.compile("([a-z]+)([0-9]+)")
                p3 = re.compile("(£)([a-z]+)")
                p4 = re.compile("([a-z]+)(£[0-9]+)")
                p5 = re.compile("([0-9]+)([a-z]+)([0-9]+)")
                p6 = re.compile("([0-9]+)(@x@[a-z]+_0)")
                p7 = re.compile("(£[0-9]+)([a-z]+)")
                if p0.match(word) and word != (m0.group(0))) {
                    String var = m0.group(0);
                    String realValue = delexicalizedMap.get(var);
                    realValue = w.replace(var, realValue);
                    delexicalizedMap.put(var, realValue);
                    observedWordSequence.add(var.trim());
                } else if (m1.matches() && m1.group(2).trim().equals("o")) {
                    observedWordSequence.add(m1.group(1).trim() + "0");
                } else if (m1.matches()) {
                    observedWordSequence.add(m1.group(1).trim());
                    observedWordSequence.add(m1.group(2).trim());
                } else if (m2.matches() && (m2.group(1).trim().equals("l") || m2.group(1).trim().equals("e"))) {
                    observedWordSequence.add("£" + m2.group(2).trim());
                } else if (m2.matches()) {
                    observedWordSequence.add(m2.group(1).trim());
                    observedWordSequence.add(m2.group(2).trim());
                } else if (m3.matches()) {
                    observedWordSequence.add(m3.group(1).trim());
                    observedWordSequence.add(m3.group(2).trim());
                } else if (m4.matches()) {
                    observedWordSequence.add(m4.group(1).trim());
                    observedWordSequence.add(m4.group(2).trim());
                } else if (m5.matches()) {
                    observedWordSequence.add(m5.group(1).trim());
                    observedWordSequence.add(m5.group(2).trim());
                    observedWordSequence.add(m5.group(3).trim());
                } else if (m6.matches()) {
                    observedWordSequence.add(m6.group(1).trim());
                    observedWordSequence.add(m6.group(2).trim());
                } else if (m7.matches() && m7.group(2).trim().equals("o")) {
                    observedWordSequence.add(m7.group(1).trim() + "0");
                } else {
                    observedWordSequence.add(w.trim());
                }
                '''
                observedWordSequence.append(word.strip())

            MR = MeaningRepresentation(self.singlePredicate, attributeValues, MRPart, delexicalizedMap)

            # We store the maximum observed word sequence length, to use as a limit during generation
            if len(observedWordSequence) > self.maxWordSequenceLength:
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

            DI = DatasetInstance(MR, directReferenceSequence, self.postProcessRef(MR, directReferenceSequence))
            self.datasetInstances[self.singlePredicate].append(DI)

    def postProcessRef(self, mr, refSeq):
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


if __name__ == '__main__':
    parser = DatasetParser("trainset.csv")
