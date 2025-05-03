class blackbox_fcn():
    r"""black-box function for BayesOpt. 
    Note that we maximize the function within this class to make it compatible with BayesOpt
    So put the correct sign when you implement evaluate_true.
    """
    def __init__(self):
        pass

    def evaluate_true(self, Z, T):
        raise NotImplementedError
