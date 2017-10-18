# This is a basic definition of the state. But it can be overriden/made more complicated to support:
# - bookkeeping on top of the actions to facilitate feature extraction, e.g. how many times a tag has been used
# - non-trivial conversion of the state to the final prediction

class State(object):

    def __init__(self):
        self.currentStageNo = -1
        self.currentStages = []

    #  updates the state with an action 
    def updateWithAction(self, action, structuredInstance):
        # call the function for the right stage
        self.currentStages[self.currentStageNo].updateWithAction(self, action, structuredInstance)
