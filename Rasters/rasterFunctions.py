import numpy as np
import matplotlib.pyplot as plt

def determineCellType(File):
    cellType = (File[-3:])
    numNeurons = 0
    if (cellType =="grr"):      # Sets numNeurons according to neuron type  
        numNeurons = 2 ** 20
        numNeurons = 4000       # Have to look at fewer granule cells because run out of memory if too many
    elif(cellType =='gor'):
        numNeurons = 4096
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


def split_raster_to_trials (raster):
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


def process_trial(trial, numNeurons, maxNumSpikes):
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


def plot_rasters(rasters, numNeurons, showNeuron):
    # check if any specific neurons are requested
    # can be single value or list of neurons
    if showNeuron != 0:
        for neuron in showNeuron:
            plotarray = rasters[:,neuron,:]
            plt.figure(figsize=(15, 6))
            plt.eventplot(plotarray, colors='black', orientation='horizontal')
            plt.title(f"Neuron: {neuron}")
            plt.ylabel("Trial")
            plt.xlabel("Timebin")
            plt.show()
    # otherwise, plot all neurons
    else:
        for neuron in range(numNeurons):
                plotarray = rasters[:,neuron,:]
                plt.figure(figsize=(15, 6))
                plt.eventplot(plotarray, colors='black', orientation='vertical')
                plt.title(f"Neuron: {neuron}")
                plt.ylabel("Timebin")
                plt.xlabel("Trial")
                plt.show()