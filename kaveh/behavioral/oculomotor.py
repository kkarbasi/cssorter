"""
Laboratory for Computational Motor Control, Johns Hopkins School of Medicine

Author: Kaveh Karbasi <kkarbasi@berkeley.edu>
"""

import numpy as np
import scipy.signal
from sklearn.cluster import KMeans

class target:
    """ Class for handling target data"""
    def __init__(self, vt, ht, dt, mode):
        """
        Object constructor
        m sklearn.cluster import KMeans
        vt: vertical target position signal
        ht: horizontal target position signal
        dt: sampling period
        """
        valid_modes = {'horizontal','vertical','2d'}
        if mode not in valid_modes:
            raise ValueError("Mode must be one of {}".format(valid_modes))
        if mode == 'horizontal':
            self.ht = ht
        if mode == 'vertical':
            self.vt = vt
        if mode == '2d':
            self.ht = ht
            self.vt = vt
        self.dt = dt
        self.mode = mode
    
    def _find_target_jumps(self):
        if self.mode == 'horizontal':
            return self._find_target_jumps_horizontal()
        if self.mode == 'vertical':
            return self._find_target_jumps_vertical()
        if self.mode == '2d':
            return self._find_target_jumps_2d()

    
    def _find_target_jumps_horizontal(self):
        """
        Finds the target jump indices in the input target horizontal position signal
        """
        # find target jumps
        ht_diff = np.abs(np.diff(self.ht))
        target_jump_indices = scipy.signal.find_peaks(ht_diff, prominence=200)[0]

        # remove detected target jumps that are sequential (less than 5 samples apart)
        to_delete = []
        for i, tji in enumerate(target_jump_indices[1:]):
                if tji - target_jump_indices[i] < 5:
                            to_delete = to_delete + [i+1]
        mask = np.ones(target_jump_indices.shape, dtype=bool)
        mask[to_delete] = False
        target_jump_indices = target_jump_indices[mask]
        return target_jump_indices
        
    
    def _find_target_jumps_vertical(self):
        """
        Finds the target jump indices in the input target vertical position signal
        """
        # find target jumps
        vt_diff = np.abs(np.diff(self.vt))
        target_jump_indices = scipy.signal.find_peaks(vt_diff, prominence=200)[0]

        # remove detected target jumps that are likely noise related(less than 5 samples apart)
        to_delete = []
        for i, tji in enumerate(target_jump_indices[1:]):
                if tji - target_jump_indices[i] < 5:
                            to_delete = to_delete + [i+1]
        mask = np.ones(target_jump_indices.shape, dtype=bool)
        mask[to_delete] = False
        target_jump_indices = target_jump_indices[mask]
        return target_jump_indices
    
    def _find_target_jumps_2d(self):
        pos_2d = np.column_stack((self.vt, self.ht))
        pos_norm = np.linalg.norm(pos_2d, axis = -1)
        pos_diff = np.abs(np.diff(pos_norm))
        target_jump_indices = scipy.signal.find_peaks(pos_diff, prominence=200)[0]

        to_delete = []
        for i, tji in enumerate(target_jump_indices[1:]):
            if tji - target_jump_indices[i] < 5:
                to_delete = to_delete + [i+1]
        mask = np.ones(target_jump_indices.shape, dtype=bool)
        mask[to_delete] = False
        target_jump_indices = target_jump_indices[mask]

        return target_jump_indices

    def _find_jump_vector_amplitudes(self, num_clusters):
        if self.mode == 'horizontal':
            return self._find_jump_vector_amplitudes_h(num_clusters)
        if self.mode == 'vertical':
            return self._find_jump_vector_amplitudes_v(num_clusters)
        if self.mode == '2d':
            return self._find_jump_vector_amplitudes_2d(num_clusters)

    def _find_jump_vector_amplitudes_h(self, num_clusters):
        target_jump_indices = self._find_target_jumps()
        self.jump_vecs = []
        for tji in target_jump_indices:
                self.jump_vecs = self.jump_vecs + [self.ht[tji + 5] - self.ht[tji - 5]]
        #[hist, bin_edges] = np.histogram(jump_vecs, bins=np.arange(np.min(self.ht), np.max(self.ht), bin_size))
        #hist[hist < 10] = 0 # remove rare target jump vectors
        #return bin_edges[np.nonzero(hist)]
        self.jump_vecs = np.array(self.jump_vecs).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_clusters, n_init = 20, n_jobs=5).fit(self.jump_vecs)
        jump_amps = kmeans.cluster_centers_
        jump_amps = np.array([int(ja) for ja in jump_amps])
        return jump_amps

    def _find_jump_vector_amplitudes_v(self, num_clusters):
        target_jump_indices = self._find_target_jumps()
        self.jump_vecs = []
        for tji in target_jump_indices:
                self.jump_vecs = self.jump_vecs + [self.vt[tji + 5] - self.vt[tji - 5]]
        #[hist, bin_edges] = np.histogram(jump_vecs, bins=np.arange(np.min(self.ht), np.max(self.ht), bin_size))
        #hist[hist < 10] = 0 # remove rare target jump vectors
        #return bin_edges[np.nonzero(hist)]
        self.jump_vecs = np.array(self.jump_vecs).reshape(-1,1)
        kmeans = KMeans(n_clusters=num_clusters, n_init = 20, n_jobs=5).fit(self.jump_vecs)
        jump_amps = kmeans.cluster_centers_
        jump_amps = np.array([int(ja) for ja in jump_amps])
        return jump_amps

    def _find_jump_vector_amplitudes_2d(self, num_clusters):
        """
        2d clustering of the jump vectors.
        """
        target_jump_indices = self._find_target_jumps();
        
        jump_vecs_h = []
        for tji in target_jump_indices:
            jump_vecs_h = jump_vecs_h + [self.ht[tji + 5] - self.ht[tji - 5]]
        jump_vecs_h = np.array(jump_vecs_h)    
                    
        jump_vecs_v = []
        for tji in target_jump_indices:
            jump_vecs_v = jump_vecs_v + [self.vt[tji + 5] - self.vt[tji - 5]]
        jump_vecs_v = np.array(jump_vecs_v)

        self.jump_vecs = np.column_stack((jump_vecs_h, jump_vecs_v))
        kmeans = KMeans(n_clusters=num_clusters, n_init = 20, n_jobs=5).fit(self.jump_vecs)
        jump_amps = kmeans.cluster_centers_

        return np.int64(jump_amps)

    def _is_in_cluster(self, jump_vec, jump_amp, jump_tol):
        if self.mode == 'horizontal':
            if jump_vec < jump_amp + jump_tol and jump_vec >= jump_amp - jump_tol:
                return True
            else:
                return False
        if self.mode == 'vertical':
            if jump_vec < jump_amp + jump_tol and jump_vec >= jump_amp - jump_tol:
                return True
            else:
                return False
        if self.mode == '2d':
            if (jump_vec[0] < jump_amp[0] + jump_tol and
                jump_vec[0] >= jump_amp[0] - jump_tol and
                jump_vec[1] < jump_amp[1] + jump_tol and
                jump_vec[1] >= jump_amp[1] - jump_tol):
                return True
            else:
                return False

    def get_target_jumps(self, num_clusters = 3, jump_tol = 100):
        """
        Returns a dictionary containing the indices of the jumps to the found jump vectors.
        The jump vectors are found by detecting all jumps, then using kmeans with k=num_clusters
        (should be determined by the experimental setup, auto detection later), then assigning each
        jump to one cluster if the euclidean distance is less than jump_tol.
        """
        jump_amps = self._find_jump_vector_amplitudes(num_clusters)

        target_jumps_to = {}
        for ja in jump_amps:
                target_jumps_to[str(ja)] = np.array([], dtype='int64')
        target_jump_indices = self._find_target_jumps()
        for i, tji in enumerate(target_jump_indices):
                #     jump_vec = ht.data[prange][tji + 5] - ht.data[prange][tji - 5]
                for ja in jump_amps:
                    if self._is_in_cluster(self.jump_vecs[i], ja, jump_tol):
                        target_jumps_to[str(ja)] = np.concatenate((target_jumps_to[str(ja)], [tji]))
        return [target_jumps_to, jump_amps]


