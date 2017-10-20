from .state import State

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from structuredPredictionNLG.Action import Action

import random

class ImitationLearner(object):

    def __init__(self, model, content_model, word2index, index2word, stateType):
        self.model = model
        self.content_model = content_model
        self.word2index = word2index
        self.index2word = index2word
        self.stateType = stateType

    # this function predicts an instance given the state
    # state keeps track the various actions taken
    # it does not change the instance in any way,
    # it does change the state
    # the predicted structured output is returned in the end
    #@profile
    def predict(self, structuredInstance, state=None, optimalPolicyProb=0.0):
        if state== None:
            state = self.stateType()

        # if we haven't started predicting, initialize the state for prediction
        '''
        if state.currentStageNo < 0:
            state.currentStageNo = 0
            state.currentStages = [self.stages[state.currentStageNo](state, structuredInstance)]
        '''

        # predict all remainins actions
        # if we do not have any actions we are done
        while len(state.agenda) > 0:
            # update the RNN hidden state to take into account the previous
            # action taken, regardless of whether it came from the expert or
            # from the RNN
            input_word, hidden_state = state.getRNNFeatures()
            index = self.word2index[input_word.label]
            action_probs, new_state = self.model.predict(index, hidden_state)
            # the first condition is to avoid un-necessary calls to random which give me reproducibility headaches
            expert_action = state.optimalPolicy(structuredInstance)
            import pdb
            pdb.set_trace()
            if (optimalPolicyProb == 1.0) or (optimalPolicyProb > 0.0 and random.random() < optimalPolicyProb):
                current_action = expert_action
                expert_action_taken = True
            else:
                # Take the model prediction
                label = action_probs.data.numpy().argmax()
                current_action = Action(self.index2word[label], state.agenda[0][0])
                expert_action_taken = False
            # add the action to the state making any necessary updates
            # TODO(kc391): should probably cache the costs so we don't have to
            # recalculate in train()
            state.updateWithAction(current_action, new_state, action_probs,
                                   expert_action, expert_action_taken,
                                   structuredInstance)

    def stateToPrediction(self, state):
        # Uncomment to print state as an action list, before it is converted to a string
        # print(state.actionsTaken)
        for act in state.actionsTaken:
            if act.label in state.datasetInstance.input.delexicalizationMap:
                act.label = state.datasetInstance.input.delexicalizationMap[act.label]
        return (" ".join([o.label for o in state.actionsTaken if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS])).strip()

    class params(object):
        def __init__(self):
            self.learningParam = 0.1
            self.iterations = 40

    #@profile
    # todo remove stages from train
    def train(self, structuredInstances):
        # for each iteration
        for iteration in range(10):
            # set the optimal policy prob
            optimalPolicyProb = pow(0.9, iteration)
            print("Iteration:"+ str(iteration) + ", optimal policy prob:"+ str(optimalPolicyProb))

            for structuredInstance in structuredInstances:

                state = self.stateType(self.content_model, structuredInstance)
                # so we obtain the predicted output and the actions taken are in state
                # this prediction uses the gold standard since we need this info for the optimal policy actions
                prediction = self.predict(structuredInstance, state, optimalPolicyProb)

                # Convert expert actions to labels
                labels = map(lambda x: self.word2index[x],
                             [action.label for action in state.expertActions])
                # see what the model predictions were and compare to the expert
                loss = self.model.fit(state.actionProbsCache, labels)
                print(loss)
                print(self.stateToPrediction(state))
                print(state.expertActions)
                print(state.actionsTaken[1:])
                print(state.expertActionsTaken)
