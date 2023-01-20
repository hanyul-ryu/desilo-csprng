
import os
import binascii
import torch
import numpy as np
import math
import chacha20_cuda
import discrete_gaussian_cuda
from discrete_gaussian_sampler import build_CDT_binary_search_tree
import randint_cuda
import randround_cuda

torch.backends.cudnn.benchmark = True

class csprng:
    def __init__(self, N=2**15, C=19, sigma=3.2,
                devices=None, seed=None, nonce=None):
        """N is the length of the polynomial, and C is the number of RNS channels.
        procure the maximum (at level zero, special multiplication) at initialization."""
        
        # By default, use all the available GPUs on the system.
        if devices is None:
            gpu_count = torch.cuda.device_count()
            self.devices = [f'cuda:{i}' for i in range(gpu_count)]
        else:
            self.devices = devices
            
        self.num_devices = len(self.devices)

        
        # If C is a number, equally distribute workloads into
        # available GPUs.
        if isinstance(C, (list, tuple)):
            self.C = sum(C)
            self.shares = C
        else:
            share_each = int(C / self.num_devices + 0.5)
            big_sharers = self.num_devices - 1
            self.shares = [share_each] * big_sharers +\
                            [C - share_each * big_sharers]
            self.C = sum(self.shares)
        
        self.N = N
        
        # We generate random bytes 4x4 = 16 per an array and hence,
        # internally only need to procure N // 4 length arrays.
        # Out of the 16, we generate discrete gaussian or uniform
        # samples 4 at a time.
        self.L = N // 4
        
        # We build binary search tree for discrete gaussian here.
        self.btree, self.btree_ptr, self.btree_size, self.tree_depth =\
            build_CDT_binary_search_tree(security_bits=128, sigma=sigma)
        
        
        
        # Counter range at each GPU.
        self.start_ind = [0] + [s * self.L for s in self.shares[:-1]]
        self.ind_increments = [s * self.L for s in self.shares]
        self.end_ind = [s + e for s, e in zip(self.start_ind, self.ind_increments)]
        
        # Total increment to add to counters after each random bytes generation.
        self.inc = self.C * self.L
                
        # expand 32-byte k.
        # This is 1634760805, 857760878, 2036477234, 1797285236.
        str2ord = lambda s : sum([2**(i*8) * c for i, c in enumerate(s)])
        self.nothing_up_my_sleeve = []
        for device in self.devices:
            str_constant = torch.tensor(
                [
                    str2ord(b'expa'), str2ord(b'nd 3'), str2ord(b'2-by'), str2ord(b'te k')
                ], device=device, dtype=torch.int64
            )
            self.nothing_up_my_sleeve.append(str_constant)
        
        # Prepare the state tensors.
        self.states = []
        for dev_id in range(self.num_devices):
            state_size = (self.shares[dev_id] * self.L, 16)
            state = torch.zeros(
                state_size, 
                dtype=torch.int64,
                device=self.devices[dev_id]
            )
            self.states.append(state)
            
        # Prepare a channeled views.
        self.channeled_states = [
            self.states[i].view(self.shares[i], self.L, -1) for i in range(self.num_devices)]
                
        # The counter.
        self.counters = []
        for dev_id in range(self.num_devices):
            ind_range = torch.arange(
                self.start_ind[dev_id],
                self.end_ind[dev_id],
                dtype=torch.int64,
                device=self.devices[dev_id]
            )
            self.counters.append(ind_range)
        
        self.refresh(seed, nonce)
    
    def refresh(self, seed=None, nonce=None, num_poly_copied=2):
        # Generate seed if necessary.
        self.key(seed)
        
        # Generate nonce if necessary.
        self.generate_nonce(nonce)
        
        # Iterate over all devices.
        for dev_id in range(self.num_devices):
            self.initialize_state_device(dev_id, seed, nonce)
            
        # We need a set of copied states for all GPUs.
        # For generation of discrete gaussian samples.
        # Pre-allocate
        self.copied_states = []
        cstate = self.states[0][:self.L * num_poly_copied].cpu()
        for dev_id in range(self.num_devices):
            self.copied_states.append(cstate.to(self.devices[dev_id]))
        self.copied_inc = self.L
        self.channeled_copied_states = [st.view(1, self.L, -1) for st in self.copied_states]
    
    def initialize_state_device(self, dev_id, seed=None, nonce=None):
                
        state = self.states[dev_id]
        state.zero_()
        
        # Set the counter.
        # It is hardly unlikely we will use CxL > 2**32.
        # Just fill in the 12th element.
        state[:, 12] = self.counters[dev_id]
        
        # Set the expand 32-bye k
        state[:, 0:4] = self.nothing_up_my_sleeve[dev_id]
        
        # Set the seed.
        state[:, 4:12] = self.seeds[dev_id]
    
        # Fill in nonce.
        state[:, 14:] = self.nonces[dev_id]
            
    
    def key(self, seed=None):
        # 256bits seed as a key.
        if seed is None:
            # 256bits key as a seed,
            nbytes = 32
            part_bytes = 4
            n_keys = nbytes // part_bytes
            hex2int = lambda x, nbytes : int(binascii.hexlify(x), 16)
            
            self.seeds = []
            for dev_id in range(self.num_devices):
                cuda_seed = torch.tensor(
                    [
                        hex2int(os.urandom(part_bytes), part_bytes) for _ in range(n_keys)
                    ],
                    dtype=torch.int64,
                    device=self.devices[dev_id]
                )
                self.seeds.append(cuda_seed)
        else:
            self.seeds = []
            for dev_id in range(self.num_devices):
                cuda_seed = torch.tensor(
                    seed[dev_id],
                    dtype=torch.int64,
                    device=self.device
                )
                self.seeds.append(cuda_seed)
    
    
    def generate_nonce(self, nonce):
        # nonce is 64bits.
        if nonce is None:
            # 256bits key as a seed,
            nbytes = 8
            part_bytes = 4
            n_keys = nbytes // part_bytes
            hex2int = lambda x, nbytes : int(binascii.hexlify(x), 16)
            
            self.nonces = []
            for dev_id in range(self.num_devices):
                cuda_nonce = torch.tensor(
                    [
                        hex2int(os.urandom(part_bytes), part_bytes) for _ in range(n_keys)
                    ],
                    dtype=torch.int64,
                    device=self.devices[dev_id]
                )
                self.nonces.append(cuda_nonce)
        else:
            self.seeds = []
            for dev_id in range(self.num_devices):
                cuda_seed = torch.tensor(
                    seed[dev_id],
                    dtype=torch.int64,
                    device=self.device
                )
                self.nonces.append(cuda_seed)

    def randbytes(self, C=None, L=None, reshape=False):
        """C is the required number of channels, smaller than self.C.
        L directly sets the length of the states."""
        
        if C is None:
            channel_ranges = self.shares
        elif isinstance(C, (list, tuple)):
            channel_ranges = C
        else:
            channel_ranges = [C] * self.num_devices
            
        if L is None:
            L = self.L            
            
        new_ranges = [C * L for C in channel_ranges]    
        ranged = [s[:r, :] for s, r in zip(self.states, new_ranges)]
        random_bytes = chacha20_cuda.chacha20(ranged, self.inc)
        
        if reshape:
            random_bytes = [rb.view(C, L, 16) for rb, C in zip(random_bytes, channel_ranges)] 
        
        return random_bytes
    
    
    def randint(self, amax, shift=0):
        """amax must be specified by channels."""
        assert isinstance(amax, (list, tuple)), "amax must be a list or a tuple."
        
        channel_ranges = [len(q) for q in amax]
        len_amax = len(amax)
        
        q_conti = [np.ascontiguousarray(q, dtype=np.uint64) for q in amax]
        q_ptr = [q.__array_interface__['data'][0] for q in q_conti]
                
        states = [
            self.channeled_states[dev_id][:channel_ranges[dev_id]]
            for dev_id in range(len_amax)]
            
        result = randint_cuda.randint_fast(states, q_ptr, shift, self.inc)
        
        return result
    
    def randint_copied(self, amax, shift=0):
        q = [[amax]] * self.num_devices
        
        q_conti = [np.ascontiguousarray(qi, dtype=np.uint64) for qi in q]
        q_ptr = [q.__array_interface__['data'][0] for q in q_conti]

        result = randint_cuda.randint_fast(
            self.channeled_copied_states, q_ptr, shift, self.copied_inc)
        
        return result
    
    def discrete_gaussian(self):
        """Generate as many samples as possible."""
        result = discrete_gaussian_cuda.discrete_gaussian_fast(self.states,
                                                              self.btree_ptr,
                                                              self.btree_size,
                                                              self.tree_depth,
                                                              self.inc)
    
    def discrete_gaussian_copied(self, num_poly=1):
        """Generates THE SAME gaussian samples for each GPU."""
        states = [st[:self.L*num_poly] for st in self.copied_states]
        result = discrete_gaussian_cuda.discrete_gaussian_fast(states,
                                                              self.btree_ptr,
                                                              self.btree_size,
                                                              self.tree_depth,
                                                              self.copied_inc)
        result = [r.view(num_poly, -1) for r in result]
        return result
    
    def randround(self, coef):
        """Randomly round coef. Coef must be a double tensor.
        coef must reside in the fist GPU in the GPUs list"""
        
        L = self.N // 16
        rand_bytes = chacha20_cuda.chacha20((self.states[0][:L],), self.inc)[0].ravel()
        randround_cuda.randround((coef, ), (rand_bytes,))
        return rand_bytes
    
