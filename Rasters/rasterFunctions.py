import numpy as np
import matplotlib.pyplot as plt

def determineCellType(File):
    cellType = (File[len(File)-3:len(File)])
    numNeurons = 0
    if (cellType =="grr"):      # Sets numNeurons according to neuron type  
        numNeurons = 2 ** 20
        numNeurons = 4000       # Have to look at fewer granule cells because run out of memory if too many
    elif (cellType =="mfr"):
        numNeurons = 4096
    elif (cellType =="ncr"):
        numNeurons = 8
    elif (cellType =="pcr"):
        numNeurons = 32
    elif (cellType =="bcr"):
        numNeurons = 128
    elif (cellType =="scr"):
        numNeurons = 512
    elif (cellType =="ior"):
        numNeurons = 4
    return cellType, numNeurons