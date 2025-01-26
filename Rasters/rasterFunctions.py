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


def splitRasterToTrials (raster):
    # using list because faster dynamic resizing
    # casted as np array for output
    bytrial_array = [] # final output
    current_trial = [] # temp list for each trial
    i = 0
    while i < len(raster):
        if raster[i] == -2:
            # check that isnt initial -2
            if current_trial:
                bytrial_array.append(np.array(current_trial))
                current_trial = []
            i += 2 # skip the following number
        else:
            current_trial.append(raster[i])
            i += 1
    # add the last trial
    if current_trial:
        bytrial_array.append(np.array(current_trial))
    # save as np.array
    bytrial_array = np.array(bytrial_array, dtype=object)
    print(bytrial_array)
    return bytrial_array

# testing
arr = np.array([-2, 0, 1, 2, 3, -2, 1, 4, 5, 6, -2, 2, 7, 8, 9, 10])
test = splitRasterToTrials(arr)

def processTrial(trial):
    return trial


