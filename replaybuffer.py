import collections
import random
import torch
import numpy as np

buffer_limit  = 10000000

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
#        print("buffer : ", self.buffer)

    def sample(self, n):
        initial_batch = self.buffer[0]
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = initial_batch
        s_lst = s_lst.reshape(1, 8)
        r_lst = r_lst.reshape(1, 1)
        a_lst = a_lst.reshape(1, 1)
        s_prime_lst = s_prime_lst.reshape(1, 8)
        done_mask_lst = done_mask_lst.reshape(1, 1)

        mini_batch = random.sample(self.buffer, n - 1)
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s = s.reshape(1, 8)
            a = a.reshape(1, 1)
            r = r.reshape(1, 1)
            s_prime = s_prime.reshape(1, 8)
            done_mask = done_mask.reshape(1, 1)
            s_lst = torch.cat([s_lst, s])
            a_lst = torch.cat([a_lst, a])
            r_lst = torch.cat([r_lst, r])
            s_prime_lst = torch.cat([s_prime_lst, s_prime])
            done_mask_lst = torch.cat([done_mask_lst, done_mask])

        return s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst

    def size(self):
        return len(self.buffer)