
from .state import State

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import random

class ImitationLearner(object):

    # this is to be specified by the class that inherits this.
    # might be worth turning it into a function that initializes the stages based on data?
    stages = None
    stateType = None

    def __init__(self):
        self.stageNo2model = []
        self.stageNo2vectorizer = []
        self.stageNo2labelEncoder = []
        for stage in self.stages:
            # we might want to have this specified by the stage to allow picking different classifiers at will
            self.stageNo2model.append((SGDClassifier(average=True)))
            self.stageNo2vectorizer.append(DictVectorizer())
            self.stageNo2labelEncoder.append(LabelEncoder())

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
        if state.currentStageNo < 0:
            state.currentStageNo = 0
            state.currentStages = [self.stages[state.currentStageNo](state, structuredInstance)]
            
        # While we are not done (do-while loop with a break)
        while True:
            # predict all remainins actions
            # if we do not have any actions we are done
            while len(state.currentStages[state.currentStageNo].agenda) > 0:
                # for each action
                # pop it from the queue                                             
                current_action = state.currentStages[state.currentStageNo].agenda.popleft()
                # extract features and add them to the action 
                # (even for the optimal policy, it doesn't need the features but they are needed later on)
                current_action.features = state.currentStages[state.currentStageNo].extractFeatures(state, structuredInstance, current_action)
                # the first condition is to avoid un-necessary calls to random which give me reproducibility headaches
                if (optimalPolicyProb == 1.0) or (optimalPolicyProb > 0.0 and random.random() < optimalPolicyProb):
                    current_action.label = state.currentStages[state.currentStageNo].optimalPolicy(state, structuredInstance, current_action)
                else:
                    # predict (probably makes sense to parallelize across instances)
                    # vectorize the features:
                    vectorized_features = self.stageNo2vectorizer[state.currentStageNo].transform(current_action.features)
                    # predict using the model
                    normalized_label = self.stageNo2model[state.currentStageNo].predict(vectorized_features)
                    # get the actual label (returns an array, get the first and only element)
                    current_action.label = self.stageNo2labelEncoder[state.currentStageNo].inverse_transform(normalized_label)[0]
                # add the action to the state making any necessary updates
                state.updateWithAction(current_action, structuredInstance)

            # move to the next stage if there is one left
            if state.currentStageNo + 1 < len(self.stages):
                state.currentStageNo +=1
                state.currentStages.append(self.stages[state.currentStageNo](state, structuredInstance))
            else:
                break
        # OK return the instance-levelprediction
        # Not sure where this functions belongs. Feels like the state, but then one needs to have a
        # task-specific state. This is the only place needed...
        return self.stateToPrediction(state)

    def stateToPrediction(self,state):
        pass

    class params(object):
        def __init__(self):
            self.learningParam = 0.1
            self.iterations = 40

    #@profile
    def train(self, structuredInstances, modelFileName, params):
        # for each stage create a dataset
        stageNo2trainingFeatures = []
        stageNo2trainingLabels = []
        for stage in self.stages:
            stageNo2trainingFeatures.append([])
            stageNo2trainingLabels.append([])

        # for each iteration
        for iteration in range(params.iterations):
            # set the optimal policy prob
            optimalPolicyProb = pow(1-params.learningParam, iteration)
            print("Iteration:"+ str(iteration) + ", optimal policy prob:"+ str(optimalPolicyProb))
            
            for structuredInstance in structuredInstances:

                state = self.stateType()
                # so we obtain the predicted output and the actions taken are in state
                # this prediction uses the gold standard since we need this info for the optimal policy actions
                newOutput = self.predict(structuredInstance, state, optimalPolicyProb)

                # how good is the current policy compared to the gold?
                #structuredInstance.output.compareAgainst(newOutput)
                
                stateCopy = self.stateType()
                # for each action in every stage taken in predicting the output
                for stageNo, stage in enumerate(state.currentStages):
                    # Enter the new stage, starting from 0
                    stateCopy.currentStageNo += 1
                    new_stage = self.stages[stateCopy.currentStageNo](stateCopy, structuredInstance)
                    stateCopy.currentStages.append(new_stage)
                    for action in stage.actionsTaken:
                        # DAgger just ask the expert
                        expert_action = stage.optimalPolicy(stateCopy, structuredInstance, action)
                        # if we wanted to have costs-to-go, we should assess all possible labels:
                        #print("inside imitation learner")
                        #print(stage.possibleLabels)

                        # add the labeled features to the training data
                        stageNo2trainingFeatures[stageNo].append(action.features)
                        stageNo2trainingLabels[stageNo].append(expert_action)

                         # take the original action chosen to proceed
                        stateCopy.currentStages[stateCopy.currentStageNo].agenda.popleft()
                        stateCopy.updateWithAction(action, structuredInstance)

            # OK, let's save the training data and learn some classifiers            
            for stageNo, stageInfo in enumerate(self.stages):
                print("training for stage:" + str(stageNo))
                # vectorize the training data collected
                training_data = self.stageNo2vectorizer[stageNo].fit_transform(stageNo2trainingFeatures[stageNo])
                # encode the labels
                encoded_labels = self.stageNo2labelEncoder[stageNo].fit_transform(stageNo2trainingLabels[stageNo])
                # train
                self.stageNo2model[stageNo].fit(training_data,encoded_labels)


                # TODO save with scikit learn pickles, probably following this https://stackoverflow.com/questions/24152282/saving-a-feature-vector-for-new-data-in-scikit-learn
                #if isinstance(stageInfo[1], str):
                #    modelStageFileName = modelFileName + "_" + stageInfo[0].__name__ + ":" + stageInfo[1] + "_model"
                #else:
                #    modelStageFileName = modelFileName + "_" + stageInfo[0].__name__  + "_model"
                #self.stageNo2model[stageNo].save(modelStageFileName)

                # save the data:
                #if isinstance(stageInfo[1], str):
                #    dataFileName = modelFileName + "_" + stageInfo[0].__name__ + ":" + stageInfo[1] + "_data"
                #else:
                #    dataFileName = modelFileName + "_" + stageInfo[0].__name__  + "_data"

                #dataFile = open(dataFileName, "w")
                #for instance in stageNo2training[stageNo]:
                #    dataFile.write(str(instance) + "\n")
                #dataFile.close()

    # TODO
    #def load(self, modelFileName):
    #    self.model.load(modelFileName + "/model_model")
            
