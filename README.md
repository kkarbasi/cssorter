# CSsorter: Python package for unsupervised detection of Complex Spikes
[![DockerHub](https://img.shields.io/docker/pulls/kkarbasi/cssorter.svg)](https://hub.docker.com/r/kkarbasi/cssorter)
[![DockerHub](https://img.shields.io/docker/cloud/build/kkarbasi/cssorter.svg)](https://hub.docker.com/r/kkarbasi/cssorter)

## System requirements:
The only requirement is having Docker installed on your machine.
List of the required python packages that the code uses are in [requirements.txt](https://github.com/kkarbasi/cssorter/blob/master/requirements.txt)

## Installation

### Using Docker:
You can readily use this package from it's docker image. To download and run the CSsorter docker, use the following command:
`docker run -it -p 8888:8888 --name cs_sorter -v /path/to/local/dir:/run/dmount:z kkarbasi/cssorter`

After running the above command, you can access a Jupyter notebook with sample run code from your browser at:
`http://127.0.0.1:8888`

The `/run/dmount` path inside the Docker container will be mounted to `/path/to/local/dir` on your local machine

For more information on using docker containers see [here](https://docs.docker.com/)

### Using pip command (comming soon!):

Installing the package:
`pip install cssorter`
 
Then use it as follows:
Assuming the voltage signal is loaded in a numpy vector named `voltage` and the sampling frequency is in `Fs`

`from cssorter.spikesorter import ComplexSpikeSorter`
`css = ComplexSpikeSorter(voltage, 1.0/Fs)`

`css.run()`


## Adjustable parameters

`css.num_gmm_components`: default=5; for noisy data use larger numbers 

`css.cs_num_gmm_components`: default=5;

The voltage window in which we check for CS signature. If the cell is very high frequeny, use shorter post_window
* `css.pre_window`: default=0.0002 seconds; 
* `css.post_window`: default=0.005 seconds;

`css.run() arguments`:

* `use_filtered = (True/False)`: If set to True, spike detection is performed on filtered(Sav-Golay) data.
* `spike_detection_dir = ('min'/'max')`: The direction towards which the adaptive spike detector should look for spikes.
* `align_spikes_to = ('min'/'max')`: If set to 'min' ('max') spikes are aligned to their minimum (maximum).
* `remove_overlaps = (True/False)`


