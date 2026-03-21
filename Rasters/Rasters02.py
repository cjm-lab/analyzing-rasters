import numpy as np
import os
from pathlib import Path
import rasterFunctions as rf
from joblib import Parallel, delayed
import time

## To do: count total spikes for each neuron first, that way we can allocate smaller arrays and
#           in the case of granule cells we can avoid loading cells that don't fire..
rasterFilepath = Path(input("Enter absolute filepath: "))
saveFilepath = rasterFilepath.with_suffix('.npy')   # Name of file for strored rasters, 
showNeuron = [0,1,2]                  # Use this if you want to look at rasters for one particular neurons. = 0 defaults to view in sequence
maxSpikesPerTrial = 5000        # This has to be larger than the most spikes any of the neurons would have in one trial
cellType, numNeurons = rf.determineCellType(str(rasterFilepath))


total_start = time.time()
if rasterFilepath.is_file():
    print("Reading data...")
    if ((rasterFilepath.suffix)!="grr"):
        data = np.fromfile(str(rasterFilepath),dtype=np.int16)
    else:
        data = np.fromfile(str(rasterFilepath),dtype=np.int32)
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
    print(result_rasters.shape)
    main_end = time.time()
    print(f"Time: {main_end - main_start:.2f}s")
    print(f"Saving file to: {str(saveFilepath)}")
    np.save(str(saveFilepath), result_rasters)
    total_end = time.time()
    print(f"Total Time: {total_end-total_start:.2f}s")
else: 
    print("File not found")
# plot the rasters for each neuron, or selected neurons
#rf.plot_rasters(result_rasters, numNeurons, showNeuron)
