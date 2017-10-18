# Gerasimos Lampouras, 2017:

class WordPredictor(object):
    # TODO Be tested, finalized
    def getExpertPolicyWordAction(rollInSeq, datasetInstance):
        minCost = 1.0

        availableWords = set()
        for refSeq in datasetInstance.evaluationReferences:
            for action in refSeq.sequence:
                availableWords.add(action.word)

        costVector = {}
        for word in availableWords:
            for refSeq in datasetInstance.evaluationReferences:
                minRefCost = 1.0
                minRollOutWordSeq = False
                refWindows = set()
                refWindows.add(di.directReference)
                for indirectRef in di.evaluationReferences():
                    refWindows.add(indirectRef)

                if refWindows:
                    minRollOut = ""
                    minRef = ""
                    for i in range(1, len(di.directReferenceSequence)):
                        rollInSeqCopy = []
                        for act in rollInSeq.sequence:
                            rollInSeqCopy.append(Action(act))
                        rollInSeqCopy.extend(di.directReferenceSequence[i:])

                        rollOut = datasetParser.postProcessWordSequence(di.meaningRepresentation, rollInSeqCopy)
                        print("ROLLOUT", rollOut)
                        print("REFS", refWindows)
                        refCost = LossFunction.getROUGE(rollOut, refWindows)

                        if refCost < minRefCost:
                            minRefCost = refCost
                            minRollOut = rollOut
                            minRef = refWindows.get(0)
                            minRollOutWordSeq = rollInSeqCopy

                if minRollOutWordSeq:
                    minRefCost = costBalance1 * minRefCost + costBalance2 * (minRollOutWordSeq.size() / datasetParser.getMaxWordSequenceLength())
                else:
                    minRefCost = 1.0
                if minRefCost < minCost:
                    minCost = minRefCost