from __future__ import print_function
import time
import copy
import torch
import logging
import blosc
from mpi4py import MPI
import numpy as np

from compression import w_decompress, w_compress
from ops import load_inet, ModelBuffer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Master():
    def __init__(self, comm, **kwargs):
        self.comm = comm   # get MPI communicator object
        self.world_size = comm.Get_size() # total number of processes
        self.comm_type = kwargs['comm_method']
        self._timeout_threshold = kwargs['timeout_threshold']
        self._device = kwargs['device']

    def build_model(self, args):
        self.inet = load_inet(args)
        self.nactors = args.nactors
        self.in_shape = (args.batch_size // self.nactors, 48, 8, 8)
        # buffer to receive Z
        self.buffer = ModelBuffer(self.world_size - 1, self.in_shape)
        # latency lst 
        self.latency = []
        self.recon_time = []
        self.inference_time = []
        self.fusion_time = []
    
    
    def eval_latency(self):
        def stat(x):
            if len(x) == 0:
                return [0, 0, 0, 0, 0, 0]
            x = np.asarray(x)
            return [np.median(x), np.mean(x), np.quantile(x, 0.99), np.quantile(x, 0.995), np.quantile(x, 0.999), np.std(x)]
        results = [stat(self.latency), stat(self.recon_time), stat(self.inference_time), stat(self.fusion_time)]
        data = {'latency': np.asarray(self.latency), 
                'recon': np.asarray(self.recon_time), 
                'inference': np.asarray(self.inference_time), 
                'fusion': np.asarray(self.fusion_time), 
                'workers': self.nactors}
        return results, data

    def run(self, batch_data):
        # preprare the send request
        start_time = time.time()
        for k in range(1, self.world_size):
            if k == 1:
                data = batch_data
            else:
                data = batch_data[:, k-2, ...]
            msg_send = w_compress(data)
            send_req = self.comm.isend(msg_send, dest=k, tag=80)
            send_req.wait()

        recv_reqs = []
        for k in range(1, self.world_size):
            req_k = self.comm.irecv(self.buffer.recv_buf[k-1], source=k, tag=88)
            recv_reqs.append(req_k)
            # req_k.wait()
            # logger.info('Receive from {}'.format(k))

        # z_fused = None 
        z_lst = []
        trace = np.zeros(self.nactors + 1)
        start_gather_time = time.time()
        lst = np.arange(1, self.world_size)
        end = False
        while not end:
            np.random.shuffle(lst)
            for k in lst:
                k_idx = k - 1
                if trace[k_idx] == 0:
                    status = recv_reqs[k_idx].test()
                    if status[0]: # okay
                        trace[k_idx] = 1
                        recv_data = status[1]
                        z_data = w_decompress(recv_data)
                        # logger.info('collect {}'.format(k))
                        if k == 1:
                            self.fusion_time.append(time.time() - start_gather_time)
                            z_fused = z_data
                        else:
                            self.inference_time.append(time.time() - start_gather_time)
                            z_lst.append(z_data)
                
                if (np.sum(trace) >= self.nactors and trace[0] > 0) or np.sum(trace) >= self.nactors + 1:
                    # logger.info('collect enought trace')
                    end = True
                    break
                
        if np.sum(trace) < self.nactors + 1 and trace[0] > 0:
            # logger.info("reconstruction")
            start_recon_time = time.time()
            # logger.info('Z-list {}'.format(len(z_lst)))
            if len(z_lst) > 0:
                substract = np.sum(np.stack(z_lst, axis=0), axis=0)
            else:
                substract = 0
            z_hat = z_fused * self.nactors - substract
            _ = self.inet.classifier(torch.Tensor(z_hat).to(self._device))
            self.recon_time.append(time.time() - start_recon_time)

        self.latency.append(time.time() - start_time)
        
        # # clear buffer
        self.comm.barrier()
        # self.buffer.reset()
        # logger.info('PS Done')


