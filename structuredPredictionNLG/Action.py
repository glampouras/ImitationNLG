# Gerasimos Lampouras, 2017:
from copy import copy

'''
 Action implements a single action of the NLG action sequences.
 The same structure is used for both content and word actions.
'''
class Action(object):
    # Default action identifier for the start of an ActionSequence.
    # When combined with a specific attribute, it denotes the start of that attribute 's subsequence.
    TOKEN_START = "@start@"
    # Default action identifier for the end of an ActionSequence.
    # When combined with a specific @attribute, it denotes the end of that attribute 's subsequence.
    TOKEN_END = "@end@"
    # Default action identifier for punctuation occurrences in an ActionSequence.
    TOKEN_PUNCT = "@punct@"
    # Default action identifier for variable occurrences in an ActionSequence.
    # This is usually combined with an attribute identifier and an index integer to distinguish between variables.
    TOKEN_X = "@x@"

    def __init__(self, word, attribute):
        # Main components of an action
        # In practice, attribute may be set as "attribute|value"
        # Each action consists of a word and corresponding attribute to which the word aligns to.
        self.word = word
        self.attribute = attribute

        # Collections used for tracking attribute/value pairs when generating the word sequence.
        # The first two loggers track attribute/value pairs according to the content sequence
        # (e.g. which attribute/value pairs we have already started and stopped generating words for).
        # The latter two loggers track attribute/value pairs according to the word sequence
        # (e.g. which attribute/value pairs have actually been expressed as words in the word sequence).
        # The difference exists because we may have stopped generating words for an attribute/value pairs in
        # the content sequence, without actually expressing that attribute/value pair as text.
        self.attrValuesBeforeThisTimestep_InContentSequence = set()
        self.attrValuesAfterThisTimestep_InContentSequence = set()
        self.attrValuesBeforeThisTimestep_InWordSequence = []
        self.attrValuesAfterThisTimestep_InWordSequence = []
        # This variable logs whether the attribute/value pair we are currently generating words for has been expressed
        # in the current word subsequence or not.
        self.isValueMentionedAtThisTimestep = False;

        # The prediction that led to this action (if applicable)
        # Used to attaing the alternative action probabilies for targeted exploration
        self.prediction = False;

    '''    
     Clone constructor.
     @param a Action whose values will be used to instantiate this object.
    '''
    def clone(self, action):
        self.word = copy.copy(action.word)
        self.attribute = copy.copy(action.attribute)

        if action.attrValuesBeforeThisTimestep_InContentSequence:
            self.attrValuesBeforeThisTimestep_InContentSequence = copy.copy(action.attrValuesBeforeThisTimestep_InContentSequence)

        if action.attrValuesAfterThisTimestep_InContentSequence:
            self.attrValuesAfterThisTimestep_InContentSequence = copy.copy(action.attrValuesAfterThisTimestep_InContentSequence)

        if action.attrValuesBeforeThisTimestep_InWordSequence:
            self.attrValuesBeforeThisTimestep_InWordSequence = copy.copy(action.attrValuesBeforeThisTimestep_InWordSequence)

        if action.attrValuesAfterThisTimestep_InWordSequence:
            self.attrValuesAfterThisTimestep_InWordSequence = copy.copy(action.attrValuesAfterThisTimestep_InWordSequence)

        self.isValueMentionedAtThisTimestep = action.isValueMentionedAtThisTimestep
        self.prediction = copy.copy(action.prediction)

    '''
     Quality-of-life method to set the content level loggers at the same time.
     @param attrValuesBeforeThisTimestep_InContentSequence Before this time-step content-level logger to be set.
     @param attrValuesAfterThisTimestep_InContentSequence After this time-step content-level logger to be set.
    '''
    def setContentAttrValueTracking(self, attrValuesBeforeThisTimestep_InContentSequence, attrValuesAfterThisTimestep_InContentSequence, predict):
        self.attrValuesBeforeThisTimestep_InContentSequence = copy.copy(attrValuesBeforeThisTimestep_InContentSequence)
        self.attrValuesAfterThisTimestep_InContentSequence = copy.copy(attrValuesAfterThisTimestep_InContentSequence)

        self.prediction = predict

    '''
     Quality-of-life method to set the loggers at the same time.
     @param attrValuesBeforeThisTimestep_InContentSequence Before this time-step content-level logger to be set.
     @param attrValuesAfterThisTimestep_InContentSequence After this time-step content-level logger to be set.
     @param attrValuesBeforeThisTimestep_InWordSequence Before this time-step word-level logger to be set.
     @param attrValuesAfterThisTimestep_InWordSequence After this time-step word-level logger to be set.
     @param isValueMentionedAtThisTimestep Logger concerning value of attribute/value pair currently generated.
    '''
    def setWordAttrValueTracking(self, attrValuesBeforeThisTimestep_InContentSequence, attrValuesAfterThisTimestep_InContentSequence, attrValuesBeforeThisTimestep_InWordSequence, attrValuesAfterThisTimestep_InWordSequence, isValueMentionedAtThisTimestep, predict):
        self.attrValuesBeforeThisTimestep_InContentSequence = copy.copy(attrValuesBeforeThisTimestep_InContentSequence)
        self.attrValuesAfterThisTimestep_InContentSequence = copy.copy(attrValuesAfterThisTimestep_InContentSequence)

        self.attrValuesBeforeThisTimestep_InWordSequence = copy.copy(attrValuesBeforeThisTimestep_InWordSequence)
        self.attrValuesAfterThisTimestep_InWordSequence = copy.copy(attrValuesAfterThisTimestep_InWordSequence)

        self.isValueMentionedAtThisTimestep = copy.copy(isValueMentionedAtThisTimestep)
        self.prediction = predict

    '''
     Sets all the loggers as copies of those in another Action.
     @param a The Action whose loggers to copy.
    '''
    def copyAttrValueTracking(self, action):
        if action.attrValuesBeforeThisTimestep_InContentSequence:
            self.attrValuesBeforeThisTimestep_InContentSequence = copy.copy(action.attrValuesBeforeThisTimestep_InContentSequence)
        if action.attrValuesAfterThisTimestep_InContentSequence:
            self.attrValuesAfterThisTimestep_InContentSequence = copy.copy(action.attrValuesAfterThisTimestep_InContentSequence)

        if action.attrValuesBeforeThisTimestep_InWordSequence:
            self.attrValuesBeforeThisTimestep_InWordSequence = copy.copy(action.attrValuesBeforeThisTimestep_InWordSequence)
        if action.attrValuesAfterThisTimestep_InWordSequence:
            self.attrValuesAfterThisTimestep_InWordSequence = copy.copy(action.attrValuesAfterThisTimestep_InWordSequence)

        self.isValueMentionedAtThisTimestep = copy.copy(action.isValueMentionedAtThisTimestep)
        self.prediction = copy.copy(action.predict)

    def __str__(self):
        return "A{" + self.word + "," + self.attribute + "}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.word == other.word
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(self.word)
