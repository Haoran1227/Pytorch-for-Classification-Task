import math

class EarlyStoppingCallback:

    def __init__(self, patience):
        #initialize all members you need
        self.patience = patience
        self.lowest_loss = float('Inf') # initialize lowest_loss as Inf
        self.epochs_no_improve = 0   # record how many continuous steps whose loss is not lower than lowest loss

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        if current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        if self.epochs_no_improve >= self.patience:
            return True
        else:
            return False


