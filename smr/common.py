"""
Copyright (c) 2016 David Herzfeld

Written by David J. Herzfeld <herzfeldd@gmail.com>
"""

import struct


def string_to_format(type_string):
    if 'int8' == type_string:
        return 'b'
    elif 'uint8' == type_string:
        return 'B'
    elif 'int16' == type_string:
        return 'h'
    elif 'uint16' == type_string:
        return 'H'
    elif 'int32' == type_string:
        return 'i'
    elif 'uint32' == type_string:
        return 'I'
    elif 'int64' == type_string:
        return 'l'
    elif 'uint64' == type_string:
        return 'L'
    elif 'float32' == type_string or 'float' == type_string:
        return 'f'
    elif 'float64' == type_string or 'double' == type_string:
        return 'd'
    elif 'string' == type_string:
        return 's'
    else:
        raise TypeError('Unknown type {:s}'.format(type_string))


def unpack_from_fd(fd, format):
    length = struct.calcsize(format)
    x = struct.unpack(format,  fd.read(length))
    if len(x) == 1:
        return x[0]
    return x