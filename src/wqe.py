# This should be an example of a sequence labeler
# The class should do all the task-dependent stuff

import imitation

import utils

from sacred import Experiment

ex = Experiment()


class WQE(imitation.ImitationLearner):

    # specify the stages
    # they are class variables, since any instance of WQE will have the same stages (types)
    # Might want to consider this too:
    # http: // www.python - course.eu / python3_class_and_instance_attributes.php
    stages = [utils.WordPredictor]


    def __init__(self):
        super(WQE,self).__init__()

    def stateToPrediction(self, state):
        """
        Convert the action sequence in the state to the
        actual prediction, i.e. a sequence of tags
        """
        tags = []
        for action in state.currentStages[0].actionsTaken:
            tags.append(action.label)
        return WQEOutput(tags)

    
class WQEInput(imitation.StructuredInput):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def __str__(self):
        return " ".join(self.tokens)
        

class WQEOutput(imitation.StructuredOutput):
    def __init__(self, tags):
        super().__init__()
        self.tags = tags

    def __str__(self):
        return " ".join(self.tags)

    def compareAgainst(self, other):
        if len(self.tags) != len(other.tags):
            print("ERROR: different number of tags in predicted and gold")
        
        wqe_eval_stats = WQEEvalStats()
        for i in range(len(self.tags)):
            if self.tags[i] != other.tags[i]:
                wqe_eval_stats.loss+=1
        
        wqe_eval_stats.accuracy = (len(self.tags) - wqe_eval_stats.loss) / float(len(self.tags))
        return wqe_eval_stats


class WQEEvalStats(imitation.EvalStats):
    def __init__(self):
        super().__init__()
        self.loss = 0 # number of incorrect tags
        self.accuracy = 1.0        


class WQEInstance(imitation.StructuredInstance):
    def __init__(self, tokens, tags=None):
        super().__init__()
        self.input = WQEInput(tokens)
        self.output = WQEOutput(tags)

@ex.automain
def toy_experiment():
    # load the training data!
    trainingInstances = [
        WQEInstance(["walk", "walk", "shop", "clean"], ["BAD", "GOOD", "GOOD", "GOOD"]),
        WQEInstance(["walk", "walk", "shop", "clean"], ["BAD", "BAD", "BAD", "GOOD"]),
        WQEInstance(["walk", "shop", "shop", "clean"], ["GOOD", "GOOD", "GOOD", "GOOD"])]

    testingInstances = [WQEInstance(["walk", "walk", "shop", "clean"]),
                        WQEInstance(["clean", "walk", "tennis", "walk"])]

    wqe = WQE()
    wqe.stages[0].possibleLabels = ["GOOD", "BAD"]

    # set the params
    params = WQE.params()
    # Setting this to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
    params.iterations = 2
    params.learningParam = 0.3

    wqe.train(trainingInstances, "temp", params)

    print(wqe.stageNo2labelEncoder[0].classes_)
    print(wqe.stageNo2vectorizer[0].inverse_transform(wqe.stageNo2model[0].coef_))

    print(wqe.predict(testingInstances[0]))
    print(wqe.predict(testingInstances[1]))
