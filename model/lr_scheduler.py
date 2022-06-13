# Code adapted from: https://pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/

import matplotlib.pyplot as plt
import numpy as np

class LearningRateDecay:

    def plot(self, epochs, title="Learning Rate Schedule"):
        # Compute the set of learning rates for each corresponding epoch
        lrs = [self(i) for i in epochs]
        # Plot the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):

    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
        # Store the base initial learning rate, drop factor, and epochs
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # Compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)
        return float(alpha)
