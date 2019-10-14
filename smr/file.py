"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import io
from .common import *
from .channel import Channel

class File():
    """An instance of an SMR file"""
    def __init__(self, filename):
        if isinstance(filename, str):
            self.filename = filename
            self.fd = open(filename, 'rb')
        elif isinstance(filename, io.IOBase):
            self.filename = filename.filename
            self.fd = filename

        self.fd.seek(0)
        self.system_id = unpack_from_fd(self.fd, 'h') # int16
        self.copyright = ''.join([i.decode() for i in unpack_from_fd(self.fd, '10c')]) # 10 char string
        self.creator = ''.join([i.decode() for i in unpack_from_fd(self.fd, '8c')]) # 8 string creator
        self.us_per_time = unpack_from_fd(self.fd, 'h')
        self.time_per_adc = unpack_from_fd(self.fd, 'h')
        self.file_state = unpack_from_fd(self.fd, 'h')
        self.first_channel = unpack_from_fd(self.fd, 'i') # int32
        self.num_channels = unpack_from_fd(self.fd, 'h')
        self.channel_size = unpack_from_fd(self.fd, 'h')
        self.extra_data = unpack_from_fd(self.fd, 'h')
        self.buffer_size = unpack_from_fd(self.fd, 'h')
        self.os_format = unpack_from_fd(self.fd, 'h')
        self.max_F_time = unpack_from_fd(self.fd, 'i')
        self.time_base = unpack_from_fd(self.fd, 'd')  # double
        if self.system_id < 6:
            self.time_base = 1e-6
        self.time_date = dict()
        self.time_date['detail'] = ''.join([i.decode() for i in unpack_from_fd(self.fd, '6c')])
        self.time_date['year'] = unpack_from_fd(self.fd, 'h')
        if self.system_id < 6:
            self.time_date['detail'] = 0
            self.time_date['year'] = 0
        unpack_from_fd(self.fd, '52b')  # 52 passing bytes
        self.comment = ''
        for i in range(0, 5):
            num_bytes = unpack_from_fd(self.fd, 'b')
            current_comment = ''.join([i.decode() for i in unpack_from_fd(self.fd, '79c')])
            self.comment = self.comment + current_comment[:int(num_bytes)]
        self.header_length = self.fd.tell()
        self.channels = []

    def _read_channel(self, index):
        # Otherwise, read our channel
        # print('Reading channel {} :'.format(index))
        channel = Channel(self.fd, index, self.system_id)
        if self.system_id < 6 and channel.kind != 0 and channel.kind != 2 and channel.kind != 3:
            channel.dt = channel.divide * self.us_per_time * self.time_per_adc * 1e-6
        else:
            channel.dt = channel.l_chan_dvd * self.us_per_time * (self.time_base)
        if channel.kind > 0:
            self.channels.append(channel)
        return channel

    def read_channels(self):
        for i in range(self.num_channels):
            self._read_channel(i)

    def get_channel(self, index):
        # Check our currently read channels
        for i in range(0, len(self.channels)):
            if self.channels[i].channel_number == index:
                return self.channels[i]
        channel = self._read_channel(index)
        return channel

    def __str__(self):
        x = ('SMR file: {:s}\n'.format(self.filename))
        x += ('System ID: {:d}\n'.format(self.system_id))
        x += ('Copyright: {:s}\n'.format(self.copyright))
        x += ('Creator: {:s}\n'.format(self.creator))
        x += ('uS per Time: {:d}\n'.format(self.us_per_time))
        x += ('Time per ADC: {:d}\n'.format(self.time_per_adc))
        x += ('File state {:d}\n'.format(self.file_state))
        x += ('First channel: {:d}\n'.format(self.first_channel))
        x += ('Channels: {:d}\n'.format(self.num_channels))
        x += ('Channel size: {:d}\n'.format(self.channel_size))
        x += ('OS Format: {:d}\n'.format(self.os_format))
        x += ('Time base: {:f}\n'.format(self.time_base))
        x += ('Time: {:d}, {:s}\n'.format(self.time_date['year'], self.time_date['detail']))
        x += ('Comment: {:s}\n'.format(self.comment))
        return x
