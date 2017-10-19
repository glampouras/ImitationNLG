# This is a basic definition of the state. But it can be overriden/made more complicated to support:
# - bookkeeping on top of the actions to facilitate feature extraction, e.g. how many times a tag has been used
# - non-trivial conversion of the state to the final prediction

from collections import deque
from copy import deepcopy


class State(object):

    # construct action agenda
    def __init__(self, structuredInstance=None):
        self.agenda = deque([])
        self.actionsTaken = []

    class Action(object):
        def __init__(self):
            self.label = None
            self.features = []

        def __deepcopy__(self, memo):
            newone = type(self)()
            newone.__dict__.update(self.__dict__)
            newone.features = deepcopy(self.features)
            return newone

    # extract features for current action in the agenda
    def extractFeatures(self, mrl, action):
        pass

    def optimalPolicy(self, structuredInstance, currentAction):
        pass

    def updateWithAction(self, action, structuredInstance):
        pass

    # by default each stage predicts till the very end for action costing, but different stages might choose differently
    # this is only used for costing
    def predict(self, structuredInstance, state, optimalPolicyProb, learner):
        return learner.predict(structuredInstance, state, optimalPolicyProb)

    # by default it is the same is the evaluation for each stage
    # the object returned by the predict above should have the appropriate function
    @staticmethod
    def evaluate(prediction, gold):
        # order in calling this matters
        return gold.compareAgainst(prediction)