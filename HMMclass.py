import numpy as np


class HMM:
    # De momento ni idea que tags poner
    tags = ["DET", "ADJ", "NOUN", "VERB", "ADV", "CONJ", "."]

    def __init__(self, input_sequence):
        # Los nombres tambien ni idea por que lo de sacar fotos a la pizarra no los no sabemos
        # Este es la matrix simetrica de tagsXtags

        # Input sequence
        self.w = input_sequence

        # Transition probability matrix
        self.A = np.zeros((len(self.tags), len(self.tags)))

        # Observation likelihoods
        self.B = np.zeros(len(self.tags), len(self.w))

        # Initial probability distribution over states
        self.pi = 0

    def train(self, dataset):
        # TODO
        # self.B = np.zeros() se tendria
        print("Training")


hmm = HMM()
