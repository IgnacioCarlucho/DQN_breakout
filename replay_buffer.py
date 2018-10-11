""" 
Data structure for implementing experience replay
Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import pickle

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.count_goal = 0
        self.buffer = deque()
        self.buffer_goal = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if r > 0:
            if self.count_goal < self.buffer_size: 
                self.buffer_goal.append(experience)
                self.count_goal += 1
            else:
                self.buffer_goal.popleft()
                self.buffer_goal.append(experience)
        else: 
            if self.count < self.buffer_size: 
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def size(self):
        return self.count + self.count_goal

    def sample_batch(self, batch_size):
        batch = []

        
        

        
        to_take = 0
        if self.count_goal == 0:
            to_take = 0
            # if I dont have succes goals I take all the experience from this
            if self.count < batch_size:
                batch = random.sample(self.buffer, self.count)
            else:
                batch = random.sample(list(self.buffer), batch_size)

            s_batch = [np.array(_[0]).astype(np.float32) / 255.0 for _ in batch]
            s2_batch = [np.array(_[4]).astype(np.float32) / 255.0 for _ in batch]
            a_batch = np.array([_[1] for _ in batch])
            r_batch = np.array([_[2] for _ in batch])
            t_batch = np.array([_[3] for _ in batch])

        else: 
            # if I do have experience
            if self.count_goal < batch_size/2: 
                batch_goal = random.sample(self.buffer_goal, self.count_goal)
                to_take = self.count_goal
            else:
                batch_goal = random.sample(list(self.buffer_goal), batch_size/2)    
                to_take = (batch_size/2)

            if self.count < (batch_size-to_take):
                batch = random.sample(self.buffer, self.count)
            else:
                batch = random.sample(list(self.buffer), (batch_size-to_take))

            
            s_batch = [np.array(_[0]).astype(np.float32) / 255.0 for _ in batch]
            s2_batch = [np.array(_[4]).astype(np.float32) / 255.0 for _ in batch]
            a_batch = np.array([_[1] for _ in batch])
            r_batch = np.array([_[2] for _ in batch])
            t_batch = np.array([_[3] for _ in batch])

            s_batch_g = [np.array(_[0]).astype(np.float32) / 255.0 for _ in batch_goal]
            s2_batch_g = [np.array(_[4]).astype(np.float32) / 255.0 for _ in batch_goal]
            a_batch_g = np.array([_[1] for _ in batch_goal])
            r_batch_g = np.array([_[2] for _ in batch_goal])
            t_batch_g = np.array([_[3] for _ in batch_goal])
            
        
            s_batch = np.vstack((s_batch, s_batch_g))
            a_batch = np.hstack((a_batch, a_batch_g))
            r_batch = np.hstack((r_batch, r_batch_g))
            t_batch = np.hstack((t_batch, t_batch_g))
            s2_batch = np.vstack((s2_batch, s2_batch_g))

        return s_batch, a_batch, r_batch, t_batch, s2_batch, to_take

    def clear(self):
        self.deque.clear()
        self.count = 0

    def save(self):
        print('saving the replay buffer')
        print('.')
        file = open('replay_buffer.obj', 'wb')
        print('..')
        pickle.dump(self.buffer, file)
        print('...')
        print('the replay buffer was saved succesfully')

    def load(self):
          
        try:
            filehandler = open('replay_buffer.obj', 'rb') 
            self.buffer = pickle.load(filehandler)
            self.count = len(self.buffer)
            print('the replay buffer was loaded succesfully')
        except: 
            print('there was no file to load')

