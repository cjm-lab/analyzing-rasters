import numpy as np
import matplotlib.pyplot as plt
import os
import rasterFunctions as rf
from joblib import Parallel, delayed
import time

## To do: count total spikes for each neuron first, that way we can allocate smaller arrays and
#           in the case of granule cells we can avoid loading cells that don't fire..
rasterFilename = 'MFtest08b.mfr'      # Name of the raster file from the simulation, If it's the first load for this file then it'll load
                                # If you've already loaded this file before then it'll read the loaded data from file.
saveFilename = "MFTest08bRasters"   # Name of file for strored rasters, 
showNeuron = [0,1,2]                  # Use this if you want to look at rasters for one particular neurons. = 0 defaults to view in sequence
                                # Can also be a list of neurons
maxSpikesPerTrial = 1000        # This has to be larger than the most spikes any of the neurons would have in one trial
cellType, numNeurons = rf.determineCellType(rasterFilename)

currentPath = os.getcwd()
Filename = currentPath + "\\" + saveFilename
savedFilename = Filename + ".npy"

total_start = time.time()
if os.path.exists(savedFilename): # no need to read in data again, upload and plot
    result_rasters = np.load(savedFilename)
else:  # Read in data first and save to file in format that's faster to read next time
    print("Reading data...")
    if ((rasterFilename[-3:])!="grr"):
        data = np.fromfile(rasterFilename,dtype=np.int16)
    else:
        data = np.fromfile(rasterFilename,dtype=np.int32)
    # split into by trial 1D arrays
    print("Splitting Raster to Trials...")
    split_start = time.time()
    data_byTrial = rf.split_raster_to_trials(data)
    split_end = time.time()
    print(f"Time: {split_end - split_start:.2f}s")
    # process each trial in parallel, put back together into 3D array by trial, neuron, spike
    print("Analyzing Trials...")
    main_start = time.time()
    result_rasters = Parallel(n_jobs=-1)(delayed(rf.process_trial)(vec, numNeurons, maxSpikesPerTrial) for vec in data_byTrial)
    result_rasters = np.array(result_rasters)
    main_end = time.time()
    print(f"Time: {main_end - main_start:.2f}s")
    np.save(Filename, result_rasters)
    total_end = time.time()
    print(f"Total Time: {total_end-total_start:.2f}s")
# plot the rasters for each neuron, or selected neurons
print("Plotting Rasters...")
rf.plot_rasters(result_rasters, numNeurons, showNeuron)