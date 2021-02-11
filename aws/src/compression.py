import numpy as np
import blosc

import time
import sys

'''we use different strategies for gradient compression and layer weight compression given API of mpi4py'''
def _trim_msg(msg):
    """
    msg : bytearray
        Somewhere in msg, 32 elements are 0x29. Returns the msg before that
    """
    i = msg.find(b'\x29'*32)
    if i == -1:
        raise Exception('trim_msg error; end of msg not found')
    return msg[:i]

def w_compress(w):
    assert isinstance(w, np.ndarray)
    packed_msg = blosc.pack_array(w, cname='blosclz')
    return packed_msg

def w_decompress(msg):
    # if sys.version_info[0] < 3:
    #     # Python 2.x implementation
    #     assert isinstance(msg, str)
    # else:
    #     # Python 3.x implementation
    #     assert isinstance(msg, bytes)
    weight = blosc.unpack_array(msg)
    return weight