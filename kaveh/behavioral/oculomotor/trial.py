import quantities as pq
import numpy as np

class trial:
    '''
    Class for handling trials
    '''
    def __init__(self,
                 trial_number, start_time,
                 end_time, primary_target_times,
                 corrective_target_times, cs_times,
                 HT=None, t_HT=None,
                 VT=None, t_VT=None,
                 HE=None, t_HE=None,
                 VE=None, t_VE=None,
                 signal=None, t_signal=None):
        '''
        HT, VT, HE, VE, signal: Type ndarray
        HT_t, VT_t, HE_t, VE_t, t_signal: Type Quantity ('s')
        trial_number: Integer
        start_time, end_time: In seconds (type: Quantity ('s'))
        '''
        self.HT = HT
        
        self.t_HT = t_HT
        self.VT = VT
        self.t_VT = t_VT
        self.HE = HE
        self.t_HE = t_HE
        self.VE = VE
        self.t_VE = t_VE
        self.signal = signal
        self.t_signal = t_signal
        self.trial_number = trial_number
        self.start_time = start_time
        self.end_time = end_time
        self.primary_target_times = primary_target_times
        self.corrective_target_times = corrective_target_times
        self.cs_times = cs_times
        self.detect_primary_and_corrective_target_directions()
        
        
    def detect_primary_and_corrective_target_directions(self):
        t_range = pq.quantity.Quantity(0.01, 's')
        self.primary_target_dir = np.zeros(self.primary_target_times.shape)
        for i, ptt in enumerate(self.primary_target_times):
            ht_vector = self.HT[np.where(np.logical_and(self.t_HT < (ptt + t_range), self.t_HT > (ptt - t_range)))]
            ht_vector = ht_vector[-1] - ht_vector[0]
            if (np.abs(ht_vector) < 0.2): ht_vector = 0
            vt_vector = self.VT[np.where(np.logical_and(self.t_VT < (ptt + t_range), self.t_VT > (ptt - t_range)))]
            vt_vector = vt_vector[-1] - vt_vector[0]
            if (np.abs(vt_vector) < 0.2): vt_vector = 0;

            self.primary_target_dir[i] = self._get_direction_angle(ht_vector, vt_vector)
        #                 print(i+1, get_direction_angle(ht_vector, vt_vector))
        self.corrective_target_dir = np.zeros(self.corrective_target_times.shape)
        for i, ptt in enumerate(self.corrective_target_times):
            ht_vector = self.HT[np.where(np.logical_and(self.t_HT < (ptt + t_range), self.t_HT > (ptt - t_range)))]
            ht_vector = ht_vector[-1] - ht_vector[0]
            if (np.abs(ht_vector) < 0.2): ht_vector = 0
            vt_vector = self.VT[np.where(np.logical_and(self.t_VT < (ptt + t_range), self.t_VT > (ptt - t_range)))]
            vt_vector = vt_vector[-1] - vt_vector[0]
            if (np.abs(vt_vector) < 0.2): vt_vector = 0;

            self.corrective_target_dir[i] = self._get_direction_angle(ht_vector, vt_vector)





    def _get_direction_angle(self, ht_vector, vt_vector):
        # Get reach out direction in a center-out 8 angle direction based on the sign of input
        # 
        if (ht_vector > 0 and vt_vector == 0): # 0 degrees
            return 0
        if (ht_vector > 0 and vt_vector > 0): # 45 degrees
            return 45
        if (ht_vector == 0 and vt_vector > 0): # 90 degrees
            return 90
        if (ht_vector < 0 and vt_vector > 0): # 135 degrees
            return 135
        if (ht_vector < 0 and vt_vector == 0): # 180 degrees
            return 180
        if (ht_vector <0  and vt_vector < 0): # 225 degrees
            return 225
        if (ht_vector == 0 and vt_vector < 0): # 270 degrees
            return 270
        if (ht_vector > 0 and vt_vector < 0): # 315 degrees
            return 315


