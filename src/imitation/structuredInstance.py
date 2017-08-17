class StructuredInstance(object):
    
    def __init__(self):
        self.input = None
        self.output = None


class StructuredInput(object):
    def __init__(self):
        pass


class StructuredOutput(object):

    # you should be able to take a (partial/complete) state and go to an output
    def __init__(self, state=None):
        pass

    # it must return an evalStats object with a loss
    def compareAgainst(self, other):
        pass


class EvalStats(object):
    def __init__(self):
        self.loss = 0 