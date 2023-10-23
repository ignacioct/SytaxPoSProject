import numpy as np
class HMM:
    #De momento ni idea que tags poner
    tags = ["DET","ADJ","NOUN","VERB","ADV","CONJ","."]
    def __init__(self):
        #Los nombres tambien ni idea por que lo de sacar fotos a la pizarra no los no sabemos
        #Este es la matrix simetrica de tagsXtags
        self.A = np.zeros((len(self.tags),len(self.tags)))
    def train(self,dataset):
        #TODO
        #self.B = np.zeros() se tendria
        print("Training")

hmm = HMM()