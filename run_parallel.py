from smr import File
import numpy as np
from cssorter.spikesorter import ComplexSpikeSorter
from joblib import Parallel, delayed
import multiprocessing

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
    
import fnmatch
import os
from psutil import virtual_memory
import time
import functools


def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                return
        return inner
    return decorator

@with_timeout(9600)
def processInputFile(arg):
    input_fn, output_fn = arg
    print('reading {} ...'.format(input_fn))
    smr_content = File(input_fn)
    smr_content.read_channels()
    voltage_chan = smr_content.get_channel(0)
    if voltage_chan.data.size > 0 :
	    print('processing {}...'.format(input_fn))
	    css = CimpleSpikeSorter(voltage_chan.data, voltage_chan.dt)
	    css.run()
	    with open(output_fn, 'wb') as output:
		print('writing {} ...'.format(output_fn))
		pickle.dump(css, output, pickle.HIGHEST_PROTOCOL)
    else:
	print('No data in channel for {}'.format(input_fn))

source_path = '../scratch/raw_data/'
#source_path = '/mnt/papers/Herzfeld_Nat_Neurosci_2018/raw_data/2006/Oscar/O89/'
target_path = '../scratch/auto_processed_spike_sort/'
#target_path = '/mnt/data/temp/kaveh/'

process_inputs = []

print('Recursive dir search on {}'.format(source_path))
for root, dirnames, filenames in os.walk(source_path):
    for filename in filenames:
        if filename.endswith('smr'):
            path_to_mkdir = os.path.join(target_path, os.path.relpath(root, source_path))
            if not os.path.exists(path_to_mkdir):
                os.makedirs(path_to_mkdir)
            print('Found smr file: {}'.format(os.path.join(root, filename)))
	    input_filename = os.path.join(root, filename)
            output_filename = os.path.join(path_to_mkdir, filename + '.pkl')
	    if not os.path.exists(output_filename):
	   	 process_inputs = process_inputs + [(input_filename, output_filename)]
num_cores = multiprocessing.cpu_count()
#num_cores = 6
mem = virtual_memory()
mem_total = mem.total/(1024*1024)
num_cores = mem_total/48000
print('Using {} processes based on available memory: {}MB'.format(num_cores, mem_total))

print('Using {} core...'.format(num_cores))     
for i in np.arange(0, len(process_inputs), num_cores):
	print('Running from {} to {} out of {} processes'.format(i, i+num_cores, len(process_inputs)))
	Parallel(n_jobs = num_cores, verbose=1)(map(delayed(processInputFile), process_inputs[i:i+num_cores]))
	
