"""
stage.py

A common interface for all stages.
"""

from collections import deque
from copy import deepcopy

class Stage(object):


    class Action(object):
        def __init__(self):
            self.label = None
            self.features = []

        def __deepcopy__(self, memo):
            newone = type(self)()
            newone.__dict__.update(self.__dict__)
            newone.features = deepcopy(self.features)
            return newone


    # construct action agenda
    def __init__(self, state=None, structuredInstance=None):
        self.agenda = deque([])
        self.actionsTaken = []
    
    # extract features for current action in the agenda
    def extractFeatures(self, state, mrl, action):
        pass
    
    def optimalPolicy(self, state, structuredInstance, currentAction):
        pass

    def updateWithAction(self, state, action, structuredInstance):
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