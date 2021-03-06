"""
Laboratory for Computational Motor Control, Johns Hopkins School of Medicine

Author: Kaveh Karbasi <kkarbasi@berkeley.edu>
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.signal
import scipy.fftpack
from scipy.stats import norm
import time
from matplotlib import pyplot as plt

class ComplexSpikeSorter:
    """
    Class for sorting complex spikes
    """
    def __init__(self, voltage, dt):
        """
        Object constructor
        """
        self.voltage = np.squeeze(np.array(voltage))
        self.signal_size = self.voltage.size
        self.dt = dt
        self.low_pass_filter_cutoff = 10000 #Hz
        self.high_pass_filter_cutoff = 1000 #Hz
        self.filter_order = 2
        self.num_gmm_components = 5
        self.gmm_cov_type = 'tied'
        self.pre_window = 0.0002 #s
        self.post_window = 0.005 #s
        self.minibatch_thresh = 50 #s - for spike detection: if signal length more than this, switch to minibatch GMM
        # Complex spike detection parameters:
        self.freq_range = (0, 3000) #Hz
        self.cs_num_gmm_components = 5 
        self.cs_cov_type = 'full'
        self.post_cs_pause_time = 0.0055 #s
        self.num_pca_components = 0.9 # If less than one, this would be the percent of explained variance 

    def run_spike_detection(self, use_filtered=True, remove_overlap=True,
            spike_detection_dir='min', align_spikes_to='min'):
        self._init_gmm()
        self._pre_process()
        print('Pre-process...' )
        delta = int(self.minibatch_thresh / self.dt)
        if delta >= self.signal_size:
            self._detect_spikes(use_filtered, select_clust=spike_detection_dir, align_to=align_spikes_to)
        else:
            self._detect_spikes_minibatch(use_filtered, select_clust=spike_detection_dir, align_to=align_spikes_to)
    
    def run(self, use_filtered=True, remove_overlap=True,
            spike_detection_dir='min', align_spikes_to='min'):
        start = time.time()
        self.run_spike_detection(use_filtered, remove_overlap,
                spike_detection_dir, align_spikes_to)
        print('Spike detection time = {}'.format(time.time() - start))
        start = time.time()
        self._align_spikes(use_filtered = use_filtered, remove_align_overlaps=remove_overlap)
        print('Align spikes time = {}'.format(time.time() - start))
        start = time.time()
        self._detect_cs_minibatch(batch_size=100)
        print('CS spike detection time = {}'.format(time.time() - start))
        start = time.time()
        self.cs_indices = self._remove_overlapping_complex_spikes()
        #self._realign_complex_spikes(pre = 0.0019, post = 0.002)
        #self._cs_post_process()
        print('CS post process time = {}'.format(time.time() - start))
    
    def _init_gmm(self):
        self.gmm = GaussianMixture(self.num_gmm_components,
                covariance_type = self.gmm_cov_type, warm_start=True)
        
        self.cs_gmm = GaussianMixture(self.cs_num_gmm_components,
                covariance_type = self.cs_cov_type, random_state=0, warm_start=True)
   
    def _pre_process(self):
        """
        Pre-processing on the input voltage signal:
        Apply zero-phase linear filter
        """
        self.voltage_filtered = np.copy(self.voltage)
        self.voltage_filtered = scipy.signal.savgol_filter(self.voltage_filtered, 5, 2, 1, self.dt)

    def _detect_spikes(self, use_filtered, select_clust, align_to):
        """
        Preliminary spike detection using a Gaussian Mixture Model
        Second edit: changed the detected index to be the peak of the raw signal 
        in order to help with future alignment to the peak of the spike waveforms.
        """
        if use_filtered:
            voltage_signal = self.voltage_filtered
        else:
            voltage_signal = self.voltage
        self.gmm = GaussianMixture(self.num_gmm_components,
                covariance_type = self.gmm_cov_type).fit(voltage_signal.reshape(-1,1))
        cluster_labels = self.gmm.predict(voltage_signal.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(voltage_signal.shape)
        if select_clust is 'max':
            spikes_cluster = np.argmax(self.gmm.means_)
            all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
            # Find peaks of each spike
            peak_times,_ = scipy.signal.find_peaks(voltage_signal[all_spike_indices])
            spike_indices = all_spike_indices[peak_times]
        elif select_clust is 'min':
            spikes_cluster = np.argmin(self.gmm.means_)
            all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
            # Find peaks of each spike
            peak_times,_ = scipy.signal.find_peaks(-1.0*voltage_signal[all_spike_indices])
            spike_indices = all_spike_indices[peak_times]

        spike_peaks = np.array([np.argmin(self.voltage[max(0, si - int(0.0005/self.dt)) :
             si + int(0.002/self.dt)]) for si in spike_indices])
        self.spike_indices = spike_indices + spike_peaks - int(0.0005/self.dt)
        if self.spike_indices[0] < 0:
            self.spike_indices[0] = spike_peaks[0]

    # TODO
    def _detect_spikes_from_range(self, prange, align_to):
        """
        Preliminary spike detection using a Gaussian Mixture Model, using only a range of filtered signal
        """
        peak_search_window_pre = 0.0008
        peak_search_window_post = 0.001
        voltage_signal = self.voltage_filtered[prange]

        signal_unfiltered = self.voltage[prange]
        #gmm = GaussianMixture(self.num_gmm_components,
        #        covariance_type = 'tied').fit(voltage_signal.reshape(-1,1))
        self.gmm.fit(voltage_signal.reshape(-1,1))
        cluster_labels = self.gmm.predict(voltage_signal.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(voltage_signal.shape)
        spikes_cluster = np.argmax(self.gmm.means_)
        all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
        # Find peaks of each spike
        peak_times,_ = scipy.signal.find_peaks(voltage_signal[all_spike_indices])
        spike_indices = all_spike_indices[peak_times]
        first_index = spike_indices[0]
        if align_to is 'max':
            spike_peaks = np.array([np.argmax(signal_unfiltered[max(0, si - int(peak_search_window_pre/self.dt)) : si + int(peak_search_window_post/self.dt)]) for si in spike_indices])
            spike_indices = spike_indices + spike_peaks - int(peak_search_window_pre/self.dt)
            # in case the first window is less then the 0.0005/dt
            if spike_indices[0] < 0:
                spike_indices[0] = spike_peaks[0]
        elif align_to is 'min':
            spike_peaks = np.array([np.argmin(signal_unfiltered[max(0, si - int(peak_search_window_pre/self.dt)) : si + int(peak_search_window_post/self.dt)]) for si in spike_indices])
            spike_indices = spike_indices + spike_peaks - int(peak_search_window_pre/self.dt)
            # in case the first window is less then the 0.0005/dt
            if spike_indices[0] < 0:
                spike_indices[0] = spike_peaks[0]
        else:
            ValueError('align_to argument should be min or max')
        return spike_indices

    def _detect_spikes_from_range_raw_signal(self, prange, select_clust, align_to):
        """
        Preliminary spike detection using a Gaussian Mixture Model, using only a range of raw signal
        """
        peak_search_window_pre = 0.0008
        peak_search_window_post = 0.001
        #voltage_signal = self.voltage_filtered[prange]
        voltage_signal = self.voltage[prange]
        signal_unfiltered = self.voltage[prange]
        #gmm = GaussianMixture(self.num_gmm_components,
        #        covariance_type = 'tied').fit(voltage_signal.reshape(-1,1))
        self.gmm.fit(voltage_signal.reshape(-1,1))
        cluster_labels = self.gmm.predict(voltage_signal.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(voltage_signal.shape)
        if select_clust is 'max':
            spikes_cluster = np.argmax(self.gmm.means_)
            all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
            peak_times,_ = scipy.signal.find_peaks(voltage_signal[all_spike_indices])
            spike_indices = all_spike_indices[peak_times]
        elif select_clust is 'min':
            spikes_cluster = np.argmin(self.gmm.means_)
            all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
            peak_times,_ = scipy.signal.find_peaks(-1.0*voltage_signal[all_spike_indices])
            spike_indices = all_spike_indices[peak_times]
        else:
            ValueError('select_clust argument should be min or max')
        
        if align_to is 'max':
            spike_peaks = np.array([np.argmax(signal_unfiltered[max(0, si - int(peak_search_window_pre/self.dt)) : si + int(peak_search_window_post/self.dt)]) for si in spike_indices])
            spike_indices = spike_indices + spike_peaks - int(peak_search_window_pre/self.dt)
            # in case the first window is less then the 0.0005/dt
            if spike_indices[0] < 0:
                spike_indices[0] = spike_peaks[0]
        elif align_to is 'min':
            spike_peaks = np.array([np.argmin(signal_unfiltered[max(0, si - int(peak_search_window_pre/self.dt)) : si + int(peak_search_window_post/self.dt)]) for si in spike_indices])
            spike_indices = spike_indices + spike_peaks - int(peak_search_window_pre/self.dt)
            # in case the first window is less then the 0.0005/dt
            if spike_indices[0] < 0:
                spike_indices[0] = spike_peaks[0]
        else:
            ValueError('align_to argument should be min or max')

        return spike_indices

    def _detect_spikes_minibatch(self, use_filtered, select_clust, align_to):
        """
        Loops through the entire voltage signal and detects spikes
        The loop is on each slices of the signal with size minibatch_thresh
        """
        print('Using minibatch spike detection, batch size = {}s, using filter (derivative) is {}'.format(self.minibatch_thresh, use_filtered))
        print('Using spikes'' {} for detection. Aligning them to their {}. '.format(select_clust, align_to))         
        delta = int(self.minibatch_thresh/self.dt)
        self.spike_indices = np.array([], dtype='int64')
        for i in np.arange(0, self.voltage_filtered.size - int(10/self.dt), delta):
            if use_filtered:
                curr_indices = self._detect_spikes_from_range(slice(i, i + delta), align_to = align_to) 
            else:
                curr_indices = self._detect_spikes_from_range_raw_signal(slice(i, i + delta), select_clust = select_clust, align_to=align_to) 
            curr_indices = curr_indices + i
            self.spike_indices = np.concatenate((self.spike_indices, curr_indices), axis=None)
        self.spike_indices = np.unique(self.spike_indices)	
    
    def _remove_overlapping_spike_windows(self):
        """
        Removes the spike indices that have overlapping alignment windows
        """
        to_delete = []

        pre_index = int(np.round(self.pre_window / self.dt))
        post_index = int(np.round(self.post_window / self.dt))

        #pre_index = int(np.round(0.0005 / self.dt))
        #post_index = int(np.round(0.006 / self.dt))
        
        for i, spike_index in enumerate(self.spike_indices[1:]):
            if (spike_index - pre_index) <=(self.spike_indices[i] + post_index) or self.spike_indices[i] - pre_index < 0:
                to_delete = to_delete + [i]
        if self.spike_indices[-1] + post_index >= self.voltage.size:
            to_delete = to_delete + [self.spike_indices.size - 1]
#         if self.spike_indices[0] - pre_index < 0:
#             to_delete = [0] + to_delete
        mask = np.ones(self.spike_indices.shape, dtype=bool)
        mask[to_delete] = False
        no_overlap_spike_indices = self.spike_indices[mask]

        return no_overlap_spike_indices

    def _remove_overlapping_complex_spikes(self):
        """
        This function is used when the overlaps are not removed before aligning the spikes.
        We first cluster all spikes (even if some of them are spikelets of the CS), then try
        merging the overlapping detected CS, with the assumption that no two CS can happen 
        less than 10ms from each other
        """
        print('Merging overlapping CS waveforms')
        csiss = np.diff(self.cs_indices)
        return np.delete(self.cs_indices, np.where(csiss < 0.010/self.dt)[0])
    
    def _realign_complex_spikes(self, pre, post):
        """
        Here we realign the detected complex spikes to their spike peak.
        Sometimes some of the detected spike indices might become misaligned
        due to _remove_overlapping_complex_spikes function. Here we try to realign them
        """
        realigned_indices = []
        Fs = 1.0/self.dt
        for i in self.cs_indices:
            realigned_indices.append(np.argmax(self.voltage[i - int(pre*Fs) : i + int(post*Fs)])+i)
        self.cs_indices = np.array(realigned_indices)
            


    def _align_spikes(self, use_filtered, remove_align_overlaps, to_exclude = []):
        """
        Aligns the spike waveforms (from pre_window to post_windows, aligned to peak times)
        Pass to_exclude if you want to exclude from adding to the align matrix
        """
        pre_index = int(np.round(self.pre_window / self.dt))
        post_index = int(np.round(self.post_window / self.dt))
        if remove_align_overlaps:
            print('Removing overlapping spikes before alignment')
            spike_indices = self._remove_overlapping_spike_windows()
        else:
            print('Using all spikes for alignment (no overlap removal)')
            spike_indices = self.spike_indices

        if use_filtered:
            signal_size_f = self.voltage_filtered.size
            alignment = np.array([[i, self.voltage_filtered[i - pre_index : i + post_index ]] 
                for i in spike_indices if i not in to_exclude if (i + post_index) < signal_size_f
                    if (i - pre_index) >= 0])
            aligned_spikes = alignment[:,1]
            self.aligned_spikes = np.array([r for r in aligned_spikes])
            self.spike_indices = alignment[:,0]
        else:
            signal_size = self.voltage.size
            alignment = np.array([[i, self.voltage[i + pre_index : i + post_index ]] 
                for i in spike_indices if i not in to_exclude if (i + post_index) < signal_size
                    if (i - pre_index) >= 0])
            aligned_spikes = alignment[:,1]
            self.aligned_spikes = np.array([r for r in aligned_spikes])
            self.spike_indices = alignment[:,0]

    # TODO
    def _choose_num_features(self, captured_variance=0.75):
        """
        Use the number of components that captures at least captured_variance of the spike waveforms
        """
        return 0

    def _extract_features(self):
        [_,powers,_] = self._find_max_powers()
        #freq_pca = normalize(time_pca)
        #print('time pca ', pca.explained_variance_ratio_)
        return self._pca_transform(powers, self.num_pca_components)

    def _pca_transform(self, aligned_input, num_components):
        
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components = num_components)
        pca.fit(aligned_input)
        transformed = pca.transform(aligned_input)
        return transformed

    def _find_max_powers(self):
        """
        Finds and returns the maximum power of all aligned spike waveforms in a specified frequency range.
        freq_range: a tuple of frequency range boundaries (in Hz)
        """
        powers = [] 
        max_powers = []
        for wf in self.aligned_spikes:
            yf = scipy.fftpack.fft(wf)
            N = wf.size
            xf = np.linspace(0.0, 1.0 / (2.0 * self.dt), N/2)
            mask = (xf < self.freq_range[1]) & (xf >= self.freq_range[0])
            power_spectrum = 2.0/N * np.abs(yf[:N//2])
            max_powers = max_powers + [np.max(power_spectrum[mask])]
            powers.append(power_spectrum[mask])
        max_powers = np.asarray(max_powers)
        powers = np.array(powers)
        return max_powers, powers, xf[mask] 
    
    def _find_integral_powers(self):
        """
        Finds and returns the integral power of all aligned spike waveforms in a specified frequency range.
        freq_range: a tuple of frequency range boundaries (in Hz)
        """
        powers = [] 
        integral_powers = []
        for wf in self.aligned_spikes:
            yf = scipy.fftpack.fft(wf)
            N = wf.size
            xf = np.linspace(0.0, 1.0 / (2.0 * self.dt), N/2)
            mask = (xf < self.freq_range[1]) & (xf >= self.freq_range[0])
            power_spectrum = 2.0/N * np.abs(yf[:N//2])
            integral_powers = integral_powers + [np.sum(power_spectrum[mask])]
            powers.append(power_spectrum[mask])
        integral_powers = np.asarray(integral_powers)
        powers = np.array(powers)
        return integral_powers, powers, xf[mask] 

    def _cluster_spike_waveforms_by_freq(self, feature_mode = 'integral' , plot_hist = False):
        """
        Clusters the found spikes into simple and complex, using a gmm_nc component GMM
        It uses the maximum power in the lower region of the frequency spectrum of the 
        spike waveforms
        """
        if feature_mode is 'integral':
            power_feature = self._find_integral_powers()[0]
        elif feature_mode is 'max':
            power_feature = self._find_max_powers()[0]
        else:
            raise ValueError('feature_mode should be either integral or max')           
        gmm = GaussianMixture(self.cs_num_gmm_components, covariance_type = self.cs_cov_type, random_state=0).fit(power_feature.reshape(-1,1))
        cluster_labels = gmm.predict(power_feature.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(power_feature.shape)

        cs_indices = self.get_spike_indices()[cluster_labels == np.argmax(gmm.means_)]
        if plot_hist:
            plt.figure()
            # uniq = np.unique(ss.d_voltage[prang] , return_counts=True)
            x = np.arange(np.min(power_feature), np.max(power_feature), 1)
            if self.cs_cov_type == 'tied':
                gauss_mixt = np.array([p * norm.pdf(x, mu, np.sqrt(gmm.covariances_.flatten())) 
                    for mu, p in zip(gmm.means_.flatten(), gmm.weights_)])
            else:
                gauss_mixt = np.array([p * norm.pdf(x, mu, sd) 
                    for mu, sd, p in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_)])
                        
            colors = plt.cm.jet(np.linspace(0,1,len(gauss_mixt)))

            # plot histogram overlaid by gmm gaussians
            for i, gmixt in enumerate(gauss_mixt):
                    plt.plot(x, gmixt, label = 'Gaussian '+str(i), color = colors[i])

                    plt.hist(power_feature.reshape(-1,1),bins=256,density=True, color='gray')
                    plt.eventplot(gmm.means_, linelength=np.max(np.max(power_feature)))
                    plt.show()
        self.cs_indices = cs_indices
    
    def _detect_cs(self, remove_overlap):
        [_,powers,_] = self._find_max_powers()
        features = self._pca_transform(powers, self.num_pca_components)
        self.cs_indices = self._cluster_spike_by_feature(features, remove_overlaps=remove_overlap)
        

    def _cluster_spike_by_feature(self, features, slice_indices = np.array([]), remove_overlaps=False):
        """
        Clusters the found spikes into simple and complex, using a gmm_nc component GMM
        It uses the maximum power in the lower region of the frequency spectrum of the 
        spike waveforms
        """
        self.cs_gmm.fit(features)
        cluster_labels = self.cs_gmm.predict(features)
        if slice_indices.size == 0:
            cs_indices = self.get_spike_indices(remove_overlaps)[cluster_labels == np.argmax(np.mean(self.cs_gmm.means_, axis=1))]
        else:
            spike_indices = self.get_spike_indices(remove_overlaps)[slice_indices]
            cs_indices = spike_indices[cluster_labels == np.argmax(np.mean(self.cs_gmm.means_, axis=1))]
        return cs_indices

  
    def _detect_cs_minibatch(self, batch_size):
        
        print('Clustering complex spikes...')
        slice_indices = self._slice_spike_indices(slice_length=batch_size)
        [_,powers,_] = self._find_max_powers()
        cs_indices = []
        for sl in slice_indices:
            curr_powers = powers[sl,:]
            cs_indices.append(self._cluster_spike_by_feature(curr_powers, sl))
        self.cs_indices = np.hstack(cs_indices)

    def _cs_post_process(self):
        """
        Post processing for complex spikes.
        pause_time: the pause time that the complex spikes should produce in the spike train.
        Any detected complex spike that produces less pause than this is ignored
        """
        # Remove detected cs that don't produce a pause in simple spikes for pause_time
        to_delete = []
        spike_indices = self.get_spike_indices()
        for i, csi in enumerate(self.cs_indices):
            if (spike_indices[-1] != csi):
                if (spike_indices[np.squeeze(np.where(spike_indices == csi)) + 1] - csi) \
                   * self.dt < self.post_cs_pause_time:
                    to_delete = to_delete + [i]
        mask = np.ones(self.cs_indices.shape, dtype = bool)
        mask[to_delete] = False
        self.cs_indices = self.cs_indices[mask]

    def _slice_spike_indices(self, slice_length):
        """
        Slicing the detected spike indices into slice_length (s) chunks and return a list
        of the indices for each chunk. Used for minibatch complex spike detection.
        """
        delta = int(slice_length/self.dt)
        if self.signal_size > (slice_length + slice_length/2.0)/self.dt:
            print('Splitting detected spikes for CS clustering ...')
            slice_indices = []    
            countall = 0
            for i in np.arange(0, self.signal_size, delta):
                slice_indices.append(np.where(np.logical_and(self.spike_indices >= i ,  self.spike_indices < i + delta))[0])
            # If the last chunk is less than 2 minutes, merge it with the penultimate chunk:
            if slice_indices[-1].size <= (slice_length/2.0)/self.dt:
                slice_indices[-2] = np.concatenate((slice_indices[-2], slice_indices[-1]))
                del slice_indices[-1]
        else:
            slice_indices = [np.arange(0, self.spike_indices.size)]
        return slice_indices


    def recluster_complex_spikes(self, freq_range=None, gmm_nc=None, cov_type=None, plot_hist = False):
        """
        Re-run complex spike clustering with new parameters for the GMM
        """
        if freq_range is not None:
            self.freq_range = freq_range
        if gmm_nc is not None:
            self.cs_num_gmm_components = gmm_nc
        if cov_type is not None:
            self.cs_cov_type = cov_type
        self._align_spikes()
        self._cluster_spike_waveforms_by_freq(plot_hist = plot_hist)
        self._cs_post_process()

    def get_cs_spike_indices(self):
        """
        Returns the detected complex spike indices
        """
        return self.cs_indices

    def get_ss_indices(self):
        """
        Returns the detected simple spike indices
        """
        return np.setdiff1d(self.spike_indices, self.cs_indices)


    def set_spike_window(self, pre_time, post_time):
        """
        Sets the spike window for complex spike detection
        """
        self.pre_window = pre_time
        self.post_window = post_time
        self._align_spikes()

    def get_spike_indices(self, remove_overlaps=True):
        """
        Returns the detected spike indices
        remove_overlaps: If we want to remove the overlapping spikes based on the set window
        """
        if remove_overlaps:
            return self._remove_overlapping_spike_windows()
        else:
            return self.spike_indices

    def plot_spike_waveforms_average(self, **kw):
        """
        Plots the average spike wavelets of the current dataset
        """
        spikes_avg = np.mean(self.aligned_spikes, axis = 0)
        spikes_std = np.std(self.aligned_spikes, axis = 0)/np.sqrt(self.aligned_spikes.shape[0])
        x = np.arange(0,self.aligned_spikes.shape[1])
        plt.figure()
        l = plt.plot(x, spikes_avg, **kw)
        plt.fill_between(x, spikes_avg - spikes_std, spikes_avg + spikes_std, color=l[0].get_color(), alpha=0.25)
        
    def plot_spike_peaks(self, use_filtered = False , figsize = (21, 5)):
        """
        Plots the voltage signal, overlaid by spike peak times,
        overlaid by lines indicating the window around the spike peaks
        """
        if use_filtered:
            plt.figure(figsize = figsize)
            plt.plot(self.voltage_filtered)
            plt.plot(self.spike_indices, self.voltage_filtered[self.spike_indices], '.r')
        else:
            plt.figure(figsize = figsize)
            plt.plot(self.voltage)
            plt.plot(self.spike_indices, self.voltage[self.spike_indices], '.r')

    def plot_voltage(self, use_filtered = False, figsize = (21, 5)):
        """
        Plots the voltage trace
        """
        plt.figure(figsize = figsize)
        if use_filtered:
            plt.plot(np.arange(0, self.voltage_filtered.size) * self.dt, self.voltage_filtered)
            plt.title('Filtered voltage vs Time(s)')
            plt.xlabel('t(s)')
        else:
            plt.plot(np.arange(0, self.voltage.size) * self.dt, self.voltage)
            plt.title('Raw voltage vs Time(s)')
            plt.xlabel('t(s)')

    def cluster_detected_cs(self, num_clusters=8, pre_time=0.0005, post_time=0.005, align_to='min'):
        """
        Clusters the detected complex spikes based on time domain 
        """
        pre_index = int(np.round(pre_time/self.dt))
        post_index = int(np.round(post_time/self.dt))
        if align_to is 'min':
            aligned_cs = np.array([self.voltage[i - pre_index : i + post_index] for i in self.cs_indices])
        elif align_to is 'max':
            aligned_cs = []
            for csi in self.cs_indices:
                align_point = np.argmax(self.voltage[csi - pre_index : csi + post_index]) + csi-pre_index
                aligned_cs.append(self.voltage[align_point - pre_index : align_point + post_index])
            aligned_cs = np.array(aligned_cs)
        else:
            raise ValueError("align_to should be min or max")
        gmm = GaussianMixture(num_clusters, covariance_type = 'full').fit(aligned_cs)
        cluster_labels = gmm.predict(aligned_cs)
        clusters = []
        for cn in np.arange(num_clusters):
            clusters.append(aligned_cs[np.where(cluster_labels == cn)])
        return clusters, cluster_labels




        
        



