from .state import State

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from structuredPredictionNLG.Action import Action

import random
import tqdm
import sys

class ImitationLearner(object):

    def __init__(self, model, content_model, word2index, index2word, attr2index, val2index, stateType):
        self.model = model
        self.content_model = content_model
        self.word2index = word2index
        self.index2word = index2word
        self.attr2index = attr2index
        self.val2index = val2index
        self.stateType = stateType

    # this function predicts an instance given the state
    # state keeps track the various actions taken
    # it does not change the instance in any way,
    # it does change the state
    # the predicted structured output is returned in the end
    #@profile
    def predict(self, structuredInstance, state=None, optimalPolicyProb=0.0, calcRollOutCostVectors=False):
        if state == None:
            state = self.stateType()

        # if we haven't started predicting, initialize the state for prediction
        '''
        if state.currentStageNo < 0:
            state.currentStageNo = 0
            state.currentStages = [self.stages[state.currentStageNo](state, structuredInstance)]
        '''

        # predict all remainins actions
        # if we do not have any actions we are done
        initial_hidden = self.model.init_hidden()
        state.RNNState.append(initial_hidden)
        while len(state.agenda) > 0:
            # update the RNN hidden state to take into account the previous
            # action taken, regardless of whether it came from the expert or
            # from the RNN
            # TODO: state.getRNNFeatures deprecated, get input word directly
            attribute, value = state.agenda[0]
            input_word, hidden_state = state.getRNNFeatures()
            word_index = self.word2index[input_word.label]
            attr_index = self.attr2index[attribute]
            val_index = self.val2index[value]
            action_probs, new_state = self.model(word_index, attr_index, val_index, hidden_state)
            # the first condition is to avoid un-necessary calls to random which give me reproducibility headaches
            expert_action, expert_costVector = state.optimalPolicy()
            expert_action = self.convertLabelToAction(expert_action, state)
            # import pdb
            # pdb.set_trace()
            if (optimalPolicyProb == 1.0) or (optimalPolicyProb > 0.0 and random.random() < optimalPolicyProb):
                if calcRollOutCostVectors:
                    current_costVector = expert_costVector
                current_action = expert_action
                expert_action_taken = True
            else:
                # Take the model prediction
                label = action_probs.data.cpu().numpy().argmax()
                if calcRollOutCostVectors:
                    current_costVector = self.learnedPolicy_rollOut(structuredInstance, state)
                current_action = self.convertLabelToAction(self.index2word[label], state)
                expert_action_taken = False
            # add the action to the state making any necessary updates
            # TODO(kc391): should probably cache the costs so we don't have to
            # recalculate in train()
            state.updateWithAction(current_action, new_state, action_probs,
                                   expert_action, expert_action_taken,
                                   structuredInstance)

    def predict_for_evaluation(self, structuredInstance, state=None):
        self.predict(structuredInstance, state, 0.0, False)

    def learnedPolicy_rollOut(self, structuredInstance, currState):
        costVector = defaultdict(lambda: 1.0)
        for alt_label in self.word2index.keys():
            if alt_label != '@go@':
                state = self.stateType(False, False, True)
                state.clone(currState)

                input_word, hidden_state = state.getRNNFeatures()
                index = self.word2index[input_word.label]
                action_probs, new_state = self.model.predict(index, hidden_state)
                alt_action = self.convertLabelToAction(alt_label, state)

                state.updateWithAction(alt_action, new_state, action_probs,
                                       alt_action, True,
                                       structuredInstance)
                # predict all remaining actions
                # if we do not have any actions we are done
                while len(state.agenda) > 0:
                    # update the RNN hidden state to take into account the previous
                    # action taken, regardless of whether it came from the expert or
                    # from the RNN
                    input_word, hidden_state = state.getRNNFeatures()
                    index = self.word2index[input_word.label]
                    action_probs, new_state = self.model.predict(index, hidden_state)

                    # Take the model prediction
                    label = action_probs.data.numpy().argmax()
                    current_action = self.convertLabelToAction(self.index2word[label], state)

                    # add the action to the state making any necessary updates
                    # TODO(kc391): should probably cache the costs so we don't have to
                    # recalculate in train()
                    state.updateWithAction(current_action, new_state, action_probs,
                                           False, False,
                                           structuredInstance)

                rollOut = [o.label.lower() for o in state.actionsTaken if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS]
                refCost = structuredInstance.output.compareAgainst(rollOut)
                costVector[alt_label] = refCost.loss
        minCost = min(costVector.values())
        if minCost != 0.0:
            for word in costVector:
                costVector[word] = costVector[word] - minCost
        bestLabels = set([act for act in costVector if costVector[act] == 0.0])

        if Action.TOKEN_EOS in bestLabels and Action.TOKEN_SHIFT in bestLabels:
            if len(self.agenda) == 1:
                return Action.TOKEN_EOS, costVector
            else:
                return Action.TOKEN_SHIFT, costVector
        bestLabel = bestLabels.pop()
        return bestLabel, costVector

    def optimalPolicy_rollOut(self, structuredInstance, currState):
        costVector = defaultdict(lambda: 1.0)
        for alt_label in self.word2index.keys():
            if alt_label != '@go@':
                state = self.stateType(False, False, True)
                state.clone(currState)

                alt_action = self.convertLabelToAction(alt_label, state)

                state.updateWithAction(alt_action, False, False,
                                       alt_action, True,
                                       structuredInstance)
                # predict all remaining actions
                # if we do not have any actions we are done
                while len(state.agenda) > 0:
                    expert_action, expert_costVector = state.optimalPolicy()
                    expert_action = self.convertLabelToAction(expert_action, state)
                    state.updateWithAction(expert_action, False, False,
                                           expert_action, True,
                                           structuredInstance)

                rollOut = [o.label.lower() for o in state.actionsTaken if o.label != Action.TOKEN_SHIFT and o.label != Action.TOKEN_EOS]
                refCost = structuredInstance.output.compareAgainst(rollOut)
                costVector[alt_label] = refCost.loss
        minCost = min(costVector.values())
        if minCost != 0.0:
            for word in costVector:
                costVector[word] = costVector[word] - minCost
        bestLabels = set([act for act in costVector if costVector[act] == 0.0])

        if Action.TOKEN_EOS in bestLabels and Action.TOKEN_SHIFT in bestLabels:
            if len(currState.agenda) == 1:
                return Action.TOKEN_EOS, costVector
            else:
                return Action.TOKEN_SHIFT, costVector
        if Action.TOKEN_SHIFT in bestLabels:
            return Action.TOKEN_SHIFT, costVector
        bestLabel = bestLabels.pop()
        return bestLabel, costVector

    def convertLabelToAction(self, label, state):
        if label == Action.TOKEN_SHIFT:
            if len(state.agenda) == 1:
                return Action(Action.TOKEN_EOS, Action.TOKEN_EOS)
            return Action(label, state.agenda[1][0])
        elif label == Action.TOKEN_EOS:
            return Action(label, Action.TOKEN_EOS)
        else:
            return Action(label, state.agenda[0][0])

    def stateToPrediction(self, state):
        # Uncomment to print state as an action list, before it is converted to a string
        # print(state.actionsTaken)
        real = []
        for act in state.actionsTaken:
            if act.label != Action.TOKEN_SHIFT and act.label != Action.TOKEN_EOS:
                if act.label in state.datasetInstance.input.delexicalizationMap:
                    for w in state.datasetInstance.input.delexicalizationMap[act.label].split(" "):
                        real.append(w)
                else:
                    real.append(act.label)
        return real

    class params(object):
        def __init__(self):
            self.learningParam = 0.1
            self.iterations = 40

    #@profile
    # todo remove stages from train
    def train(self, structuredInstances, devStructuredInstances):
        # for each iteration
        for iteration in range(10):
            # set the optimal policy prob
            optimalPolicyProb = pow(0.9, iteration)
            print("------------------------")
            print("Iteration:" + str(iteration) + ", optimal policy prob:" + str(optimalPolicyProb))

            for structuredInstance in tqdm.tqdm(structuredInstances, ncols=120):
                state = self.stateType(self.content_model, structuredInstance, True)
                # so we obtain the predicted output and the actions taken are in state
                # this prediction uses the gold standard since we need this info for the optimal policy actions
                self.model.zero_grad()
                prediction = self.predict(structuredInstance, state, optimalPolicyProb)

                # Convert expert actions to labels
                labels = map(lambda x: self.word2index[x],
                             [action.label for action in state.expertActions])
                # see what the model predictions were and compare to the expert
                loss = self.model.fit(state.actionProbsCache, labels)
                print()
                print(loss)
                print(self.stateToPrediction(state))

            self.evaluate(devStructuredInstances, 'dev', iteration)

    def evaluate(self, structuredInstances, filename=None, epoch=None):
        if filename:
            f = open(filename, 'a')
        else:
            f = sys.stdout

        print("------------------------", file=f)
        print("Evaluating instances for epoch {}".format(epoch), file=f)

        avgBLEU = 0.0
        for structuredInstance in structuredInstances:
            state = self.stateType(self.content_model, structuredInstance, False)

            self.model.zero_grad()
            self.predict_for_evaluation(structuredInstance, state)

            realization = self.stateToPrediction(state)
            print("--------------------------------------", file=f)
            print(structuredInstance.input.MRstr)
            print(realization)
            stats = structuredInstance.output.evaluate(realization)

            avgBLEU += stats.BLEU
        avgBLEU /= len(structuredInstances)
        print("BLEU:", avgBLEU, file=f)

        f.flush()
        f.close()
