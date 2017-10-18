# Gerasimos Lampouras, 2017:
from Action import Action


# TODO Be tested, finalized
def getExpertPolicyWordAction(rollInSeq, datasetInstance):
    minCost = 1.0

    availableWords = set()
    for refSeq in datasetInstance.evaluationReferences:
        for action in refSeq.sequence:
            availableWords.add(action.word)

    costVector = {}
    for word in availableWords:
        minRefCost = 1.0
        minRollOutWordSeq = False
        refWindows = set()
        refWindows.add(datasetInstance.directReference)
        for indirectRef in datasetInstance.evaluationReferences:
            refWindows.add(indirectRef)

        if refWindows:
            minRollOut = ""
            minRef = ""
            for i in range(1, len(datasetInstance.directReferenceSequence)):
                rollInSeqCopy = []
                for act in rollInSeq.sequence:
                    rollInSeqCopy.append(Action(act))
                rollInSeqCopy.extend(datasetInstance.directReferenceSequence[i:])

                #rollOut = datasetParser.postProcessWordSequence(datasetInstance.meaningRepresentation, rollInSeqCopy)
                print("ROLLOUT", rollInSeqCopy)
                print("REFS", refWindows)
                #refCost = LossFunction.getROUGE(rollOut, refWindows)

                #if refCost < minRefCost:
                #    minRefCost = refCost
                #    minRollOut = rollOut
                #    minRef = refWindows.get(0)
                #    minRollOutWordSeq = rollInSeqCopy

        #if minRollOutWordSeq:
#                minRefCost = costBalance1 * minRefCost + costBalance2 * (minRollOutWordSeq.size() / datasetParser.getMaxWordSequenceLength())
#           else:
#              minRefCost = 1.0
        if minRefCost < minCost:
            minCost = minRefCost
        costVector[word] = minCost
