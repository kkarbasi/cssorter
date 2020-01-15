"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

from .common import *
import numpy as np

class Channel:
    """Representation of a channel"""

    def __init__(self, fd, channel_number, system_id=6):
        self.channel_number = channel_number
        self.offset = 512 + (140 * channel_number)
        fd.seek(self.offset)

        self.del_size = unpack_from_fd(fd, 'h')
        self.next_del_block = unpack_from_fd(fd, 'i')
        self.first_block = unpack_from_fd(fd, 'i')
        self.last_block = unpack_from_fd(fd, 'i')
        self.num_blocks = unpack_from_fd(fd, 'h')
        self.num_extra = unpack_from_fd(fd, 'h')
        self.pre_trigger = unpack_from_fd(fd, 'h')
        self.free_0 = unpack_from_fd(fd, 'h')
        self.physical_size = unpack_from_fd(fd, 'h')
        self.max_data = unpack_from_fd(fd, 'h')

        num_bytes = unpack_from_fd(fd, 'b')
        self.comment = ''.join([i.decode() for i in unpack_from_fd(fd, '71c')])
        self.comment = self.comment[:num_bytes]

        self.max_channel_time = unpack_from_fd(fd, 'i')
        self.l_chan_dvd = unpack_from_fd(fd, 'i')
        self.physical_channel = unpack_from_fd(fd, 'h')

        num_bytes = unpack_from_fd(fd, 'b')
        self.title = ''.join([i.decode() for i in unpack_from_fd(fd, '9c')])
        self.title = self.title[:num_bytes]

        self.ideal_rate = unpack_from_fd(fd, 'f')
        self.kind = unpack_from_fd(fd, 'B')
        unpack_from_fd(fd, 'b')  # padding
        self.interleave = True
        # print('Channel kind is {}'.format(self.kind))
        if self.kind == 1 or self.kind == 6 or self.kind == 5:
            self.scale = unpack_from_fd(fd, 'f')
            self.offset = unpack_from_fd(fd, 'f')

            num_bytes = unpack_from_fd(fd, 'b')
            self.units = ''.join([i.decode() for i in unpack_from_fd(fd, '5c')])
            self.units = self.units[:num_bytes]
            if system_id < 6:
                self.divide = unpack_from_fd(fd, 'h')
            else:
                self.interleave = unpack_from_fd(fd, 'h')
        elif self.kind == 7 or self.kind == 9:
            self.min = unpack_from_fd(fd, 'f')
            self.max = unpack_from_fd(fd, 'f')
            num_bytes = unpack_from_fd(fd, 'b')
            self.units = ''.join([i.decode() for i in unpack_from_fd(fd, '5c')])
            self.units = self.units[:num_bytes]
        elif self.kind == 4:
            self.init_low = unpack_from_fd(fd, 'B')
            self.next_high = unpack_from_fd(fd, 'B')
        # elif self.kind == 2:
        #     self.init_low = unpack_from_fd(fd, 'Q')
        # elif self.kind == 3:
        #     self.next_high = unpack_from_fd(fd, 'Q')

        self.data = []
        self.blocks = []
        if self.num_blocks > 0:
            self.blocks = [self.first_block]
            fd.seek(self.first_block)
            for i in range(self.num_blocks-1):
                # last_block, next_block, start_time, end_time = unpack_from_fd(fd, '4i'))
                _, next_block, _, _ = unpack_from_fd(fd, '4i')
                self.blocks.append(next_block)
                fd.seek(next_block)

        if self.kind == 0:
            pass  # No data, nothing to do
        elif self.kind == 1:
            self._read_adc_channel(fd)
        elif self.kind == 2 or self.kind == 3 or self.kind == 4:
            self._read_event_channel(fd)
        # elif self.kind == 6:
        #     self._read_wavemark_channel(fd)
        # elif self.kind == 5:
        #     self._read_event_channel(fd)
        else:
            print('Not implemented (type = {:d})'.format(self.kind))
            #raise RuntimeError('Unknown channel type')

    def _read_adc_channel(self, fd):
        for i in range(0, len(self.blocks)):
            fd.seek(self.blocks[i] + 18)  # offset in block header (4 bytes x 4 ints (last, next, start, end) = 16) + short channel number
            num_elements = unpack_from_fd(fd, 'h')
	    if num_elements == 1:
            	self.data += [unpack_from_fd(fd, '{:d}h'.format(num_elements))]
	    else:
                self.data += list(unpack_from_fd(fd, '{:d}h'.format(num_elements)))
        self.data = np.array(self.data, dtype='int16')

    def _read_event_channel(self, fd):
        for i in range(0, len(self.blocks)):
            fd.seek(self.blocks[i] + 18)
            num_elements = unpack_from_fd(fd, 'h')
            self.data += list(unpack_from_fd(fd, '{:d}i'.format(num_elements)))
        self.data = np.array(self.data, dtype='int32')

    def _read_wavemark_channel(self, fd):
        self.markers = []
        self.adc = []
        for i in range(0, len(self.blocks)):
            fd.seek(self.blocks[i] + 18)
            num_elements = unpack_from_fd(fd, 'h')
            self.data += list(unpack_from_fd(fd, '{:d}i'.format(num_elements)))
            self.markers += list(unpack_from_fd(fd, '{:d}BBBB'.format(num_elements)))
            self.adc += list(unpack_from_fd(fd, '{:d}h'.format(num_elements)))
        self.data = np.array(self.data, dtype='int32')

    def _read_marker_channel(self, fd):
        self.markers = []
        for i in range(0, len(self.blocks)):
            fd.seek(self.blocks[i] + 18)
            num_elements = unpack_from_fd(fd, 'h')
            self.data += list(unpack_from_fd(fd, '{:d}i'.format(num_elements)))
            self.markers += list(unpack_from_fd(fd, '{:d}BBBB'.format(num_elements)))
        self.data = np.array(self.data, dtype='int32')




    def single(self):
        """Get a numpy signle represenentation of the data"""
        return np.squeeze((self.data.astype(dtype='float32')) * (self.scale / 6553.6) + self.offset)

    def double(self):
        return np.squeeze((self.data.astype(dtype='float64')) * (self.scale / 6553.6) + self.offset)

    def __str__(self):
        x = 'Channel {:d}: {:s}\n'.format(self.channel_number, self.title)
        x += 'Comment: {:s}\n'.format(self.comment)
        x += 'Ideal rate (Hz): {:f}\n'.format(self.ideal_rate)
        if self.kind == 1 or self.kind == 6:
            x += 'Scale, Offset: {:f}, {:f}\n'.format(self.scale, self.offset)
            x += 'Units: {:s}\n'.format(self.units)
        return x
