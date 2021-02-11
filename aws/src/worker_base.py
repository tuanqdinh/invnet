from __future__ import print_function
from mpi4py import MPI
import numpy as np


import torch
from torch.autograd import Variable

import time
from datetime import datetime
import copy
import logging

from compression import w_decompress, w_compress
from ops import load_inet, ModelBuffer, load_fnet

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Worker():
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.rank = comm.Get_rank() # rank of this Worker
        self._device = kwargs['device']

    def build_model(self, args, fusion=False):
        self.inet = load_inet(args)
        self.is_fusion = fusion
        self.nactors = self.world_size - 1
        M = args.batch_size // self.nactors
        self.in_shape = (M, 3, 32, 32)

        self.buffer = ModelBuffer(1, self.in_shape)
    
    def run(self, delay):
        # receive data from server
        req = self.comm.irecv(self.buffer.recv_buf[0], source=0, tag=80)
        compressed_data = req.wait()
        data = w_decompress(compressed_data)
        with torch.no_grad():
            data = torch.Tensor(data).to(self._device)
            y, _, _ = self.inet(data)

        msg_send = w_compress(y.cpu().numpy().astype(np.float32))
        send_req = self.comm.isend(msg_send, dest=0, tag=88)
        # send_req.wait()
        self.comm.barrier()
        # self.buffer.reset()
        # logger.info('Worker {}  Done'.format(self.rank))
