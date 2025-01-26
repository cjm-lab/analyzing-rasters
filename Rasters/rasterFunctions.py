import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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
            if current_trial: # check that isnt initial -2
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
    return bytrial_array


def processTrial(trial, numNeurons, maxNumSpikes):
    bin = 0
    spikeCounter = [0] * numNeurons
    output_raster = np.zeros((numNeurons, maxNumSpikes), dtype=np.int32)
    
    i = 0
    while i < len(trial):
        if trial[i] == -1:
            bin = trial[i+1]
            i += 2 # skip bin num in next loop
        else:
            # safety checks
            if spikeCounter[trial[i]] < maxNumSpikes:
                if output_raster[trial[i], spikeCounter[trial[i]]] == 0:
                    # store the bin number sequntially in neuron indexed value
                    # NOTE right now the bin number is increased by 1 for visibility at first bin
                    output_raster[trial[i], spikeCounter[trial[i]]] = bin + 1
                    # increment the spike counter to place in next cell for same neuron
                    spikeCounter[trial[i]] += 1
                    i += 1
                else:
                    print("""ERROR: Output raster cell double counted""")
                    return IndexError
            else: 
                print("""ERROR: Neuron spiked more times than array allows,
                        increase maxNumSpikes""")
                return ReferenceError # not actually sure what error is appropriate
    return output_raster


# testing
arr = np.array(
    [-2, 0, -1, 0, 1, 2, 3, -1, 1, 4, 5, 6, 2, 1,
     -2, 1, -1, 0, 4, 5, 6, -1, 1, 7, 8, 9, 4, 6, 1,
     -2, 2, -1, 0, 7, 8, 9, -1, 1, 1, 2, 3, 7])
test = splitRasterToTrials(arr)
# Produces output of [Trial, cell, spikeBin]
result_rasters = Parallel(n_jobs=3)(delayed(processTrial)(vec, 10, 3) for vec in test)
result_rasters = np.array(result_rasters)

print(result_rasters)
print(result_rasters.shape)


# TODO figure out what needs to be plotted
for Showneuron in range(0,9):
        plotarray = result_rasters[:,Showneuron,:]
        plt.figure(figsize=(15, 6))
        plt.eventplot(plotarray, colors='black', lineoffsets=1,
                            linelengths=1)
        plt.show()