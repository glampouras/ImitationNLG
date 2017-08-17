import imitation

import utils

from sacred import Experiment

ex = Experiment()

# We first need to define the input
class NERInput(imitation.StructuredInput):
    def __init__(self, tokens):
        #super().__init__()
        self.tokens = tokens

    def __str__(self):
        return " ".join(self.tokens)


# Then the NER eval stats
class NEREvalStats(imitation.EvalStats):
    def __init__(self):
        super().__init__()
        self.FNs = 0
        self.FPs = 0
        self.TPs = 0
        self.loss = 0  # number of FPs and FNs
        self.precision = 1.0
        self.recall = 1.0
        self.F1 = 1.0


# Then the output
class NEROutput(imitation.StructuredOutput):
    # the output is tagged entities in the sentence, not just tags
    def __init__(self, tags=None):
        self.nes = set()
        self.tags = []
        if tags!=None:
            self.tags = tags
            ne = None
            # initialize the list of NEs, to be populated when the instance is built
            for i, tag in enumerate(self.tags):
                if tag.startswith("B-"):
                    # if we have already been processing a NE, add it to the list
                    if ne is not None:
                        # HACK to make the list hashable to use in the set
                        self.nes.add(str([ne, label]))
                    # start a new ne, assume no - in the type
                    label = tag.split("-")[1]
                    ne = [i]
                elif tag.startswith("I-"):
                    # a continuation
                    ne.append(i)
                elif tag == "O":
                    # do nothing unless we were on an entity
                    if ne is not None:
                        self.nes.add(str([ne, label]))
                        # initialize the ne and label again
                        ne = None
                        label = None
                label = None


    def __str__(self):
        return " ".join(self.tags) + "\n" + ",".join(self.nes)

    def compareAgainst(self, predicted):
        if len(self.tags) != len(predicted.tags):
            print("ERROR: different number of tags in predicted and gold")

        ner_eval_stats = NEREvalStats()
        ner_eval_stats.TPs = self.nes & predicted.nes
        ner_eval_stats.FPs = predicted.nes - self.nes
        ner_eval_stats.FNs = self.nes - predicted.nes
        ner_eval_stats.loss = len(ner_eval_stats.FNs) + len(ner_eval_stats.FPs)

        ner_eval_stats.precision = len(ner_eval_stats.TPs)/len(predicted.nes)
        ner_eval_stats.recall = len(ner_eval_stats.TPs)/len(self.nes)

        ner_eval_stats.F1 = (2*ner_eval_stats.precision*ner_eval_stats.recall)/\
                            (ner_eval_stats.recall+ner_eval_stats.precision)

        return ner_eval_stats


class NERInstance(imitation.StructuredInstance):
    def __init__(self, tokens, tags=None):
        super().__init__()
        self.input = NERInput(tokens)
        self.output = NEROutput(tags)


class NERState(imitation.State):
    def __init__(self):
        super().__init__()
        self.predictedNEs = set()
        self.partial_ne = None
        self.ne_label = None

# Let's inherit word predictor to adjust it to NER. Need to override the expert policy
# and the update with action to add things to the state
class NERWordPredictor(utils.WordPredictor):

    def optimalPolicy(self, state, structuredInstance, action):
        # This needs to return sometimes the label from the instance, but other times O to avoid half-entities
        # if we should start a new entity, then do it
        if structuredInstance.output.tags[action.tokenNo].startswith("B-"):
            return structuredInstance.output.tags[action.tokenNo]
        # if we are continuing an entity
        elif structuredInstance.output.tags[action.tokenNo].startswith("I-"):
            # check if we have predicted the part until now correctly:
            gold_label = structuredInstance.output.tags[action.tokenNo].split("-")[1]
            # first check the label
            if gold_label == state.ne_label:
                # then check the tokens it covers:
                gold_partial_ne = []
                token_idx = action.tokenNo
                while True:
                    token_idx -=1
                    if structuredInstance.output.tags[token_idx].startswith("I-"):
                        gold_partial_ne.append(token_idx)
                    elif structuredInstance.output.tags[token_idx].startswith("B-"):
                        gold_partial_ne.append(token_idx)
                        # reached the beginning, break!
                        break
                if gold_partial_ne == state.partial_ne:
                    return structuredInstance.output.tags[action.tokenNo]
                # in all other cases return O
                else:
                    return "O"
            else:
                return "O"
        else:
            return "O"

    def updateWithAction(self, state, action, structuredInstance):
        # add it as an action though
        self.actionsTaken.append(action)

        # update the state with (partial) NE
        if action.label.startswith("B-"):
            # if we have already been processing a NE, add it to the list
            if state.partial_ne is not None:
                # HACK to make the list hashable to use in the set
                state.predictedNEs.add(str([state.partial_ne, state.ne_label]))
            # start a new ne, assume no - in the label
            state.ne_label = action.label.split("-")[1]
            state.partial_ne = [action.tokenNo]
        elif action.label.startswith("I-"):
            # a continuation, assuming that it is legal
            # to ensure no illegal transitions we should change the predic function
            state.partial_ne.append(action.tokenNo)

        elif action.label == "O":
            # do nothing unless we were on an entity
            if state.partial_ne is not None:
                state.predictedNEs.add(str([state.partial_ne, state.ne_label]))
                # initialize the ne and label again
                ne = None
                label = None


class NERTagger(imitation.ImitationLearner):
    # specify the stages
    stages = [NERWordPredictor]
    stateType = NERState

    def __init__(self):
        super().__init__()

    def stateToPrediction(self, state):
        """
        Convert the action sequence in the state to the
        actual prediction, i.e. a sequence of tags
        """
        tags = []
        for action in state.currentStages[0].actionsTaken:
            tags.append(action.label)
        return NEROutput(tags)



@ex.automain
def toy_experiment():
    # load the training data!
    trainingInstances = [
        NERInstance(["I", "studied", "in", "London", "with", "Sebastian", "Riedel"], ["O", "O", "O", "B-LOC", "O", "B-PER", "I-PER"])]

    trainingInstances.extend(5*[NERInstance(["San", "Sebastian", "is", "great"], ["B-LOC", "I-LOC", "O", "O"])])

    testingInstances = [NERInstance(["Who", "is", "Sebastian", "Riedel"])]

    tagger = NERTagger()
    tagger.stages[0].possibleLabels = set()
    for tr in trainingInstances:
        for tag in tr.output.tags:
            tagger.stages[0].possibleLabels.add(tag)

    # set the params
    params = NERTagger.params()
    # Setting the iteratios to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
    params.iterations = 1
    params.learningParam = 0.3

    tagger.train(trainingInstances, "temp", params)

    #print(tagger.stageNo2labelEncoder[0].classes_)
    print(tagger.stageNo2vectorizer[0].inverse_transform(tagger.stageNo2model[0].coef_))

    print(tagger.predict(testingInstances[0]))
