from imitation.stage import Stage
#from collections import deque, Counter
import random
from copy import copy, deepcopy

class WordPredictor(Stage):

    class WordAction(Stage.Action):
        def __init__(self):
            super(Stage.Action,self).__init__()
            # This keeps the info needed to know which action we are taking
            self.tokenNo = -1

    # the agenda for word prediction is one action per token
    def __init__(self, state=None, structuredInstance=None):
        super(WordPredictor, self).__init__()
        # Assume 0 indexing for the tokens
        if structuredInstance == None:
            return
        for tokenNo, token in enumerate(structuredInstance.input.tokens):
            newAction = WordPredictor.Action()
            newAction.tokenNo = tokenNo
            self.agenda.append(newAction)

    def optimalPolicy(self, state, structuredInstance, action):
        # this returns the gold label for the action token as stated in the instance gold in instace.output
        return structuredInstance.output.tags[action.tokenNo]

    def updateWithAction(self, state, action, structuredInstance):
        # one could update other bits of the state too as desired.
        # add it as an action though
        self.actionsTaken.append(action)

    # all the feature engineering goes here
    def extractFeatures(self, state, structuredInstance, action):
        # e.g the word itself that we are tagging
        features = {"currentWord=" + structuredInstance.input.tokens[action.tokenNo]: 1}

        # features based on the previous predictionsof this stage are to be accessed via the self.actionsTaken
        # e.g. the previous action
        if len(self.actionsTaken)> 0:
            features["prevPrediction="+ self.actionsTaken[-1].label] = 1
        else:
            features["prevPrediction=NULL"] = 1
        
        # features based on earlier stages via the state variable.

        return features
