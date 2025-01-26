import numpy as np
import matplotlib.pyplot as plt
import os
import rasterFunctions as rf
## To do: count total spikes for each neuron first, that way we can allocate smaller arrays and
#           in the case of granule cells we can avoid loading cells that don't fire..
rasterFilename = 'm52.gor'      # Name of the raster file from the simulation, If it's the first load for this file then it'll load
                                # If you've already loaded this file before then it'll read the loaded data from file.
saveFilename = "M52goRasters"   # Name of file for strored rasters, 
Showneuron = 0                  # Use this if you want to look at rasters for one particular neurons. = 0 defaults to view in sequence

maxSpikesPerTrial = 1000         # This has to be larger than the most spikes any of the neurons would have in one trial
numTrials = 200                 # Cuttoff for number of trials analyzed       

cellType, numNeurons = rf.determineCellType(rasterFilename)
rasters = np.zeros([numNeurons,numTrials,maxSpikesPerTrial],dtype=np.int32) 
rowCounter = np.zeros((numNeurons,numTrials),dtype=np.int32) 
numSpikes = np.zeros(numTrials)
currentPath = os.getcwd()
Filename = currentPath + "\\" + saveFilename
savedFilename = Filename + ".npy"

if os.path.exists(savedFilename): # no need to read in data again
    rasters = np.load(savedFilename)
else:  # Read in data first and save to file in format that's faster to read next time
    if ((rasterFilename[len(rasterFilename)-3:len(rasterFilename)])!="grr"):
        data = np.fromfile(rasterFilename,dtype=np.int16)
    else:
        data = np.fromfile(rasterFilename,dtype=np.int32)
    counter = 0
    trial = 0
    bin = 0
    while counter < len(data): # the data are identities of neurons, we keep trial of which trial and time bin
        if data[counter] == -2:
            trial = data[counter+1]
            counter +=1  # this involves reading two values, the -2 and the trial number that follows
            print(trial)
        elif data[counter] == -1:
            bin = data[counter+1]
            counter += 1  # ditto, only -1 and the bin number that follows...
        else:
            if data[counter]<numNeurons:
                if rowCounter[data[counter],trial]<maxSpikesPerTrial:
                    rasters[data[counter],trial, rowCounter[data[counter],trial]] = bin
                    rowCounter[data[counter],trial]+=1
        counter += 1
        if trial >= numTrials-1:
            break
    np.save(Filename, rasters)

colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 1
if Showneuron != 0:
    plotarray = rasters[Showneuron,:,:]
    plt.figure(figsize=(15, 6))
    plt.eventplot(plotarray, colors=colors2, lineoffsets=lineoffsets2,
                        linelengths=linelengths2)
    plt.show()  
else:  
    print(numNeurons)
    for Showneuron in range(0,numNeurons):
        plotarray = rasters[Showneuron,:,:]
        plt.figure(figsize=(15, 6))
        plt.eventplot(plotarray, colors=colors2, lineoffsets=lineoffsets2,
                            linelengths=linelengths2)
        plt.show()

