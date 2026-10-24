#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import logging
import os

import torch #type: ignore
import torch.nn.functional as F #type: ignore
import torch.optim as optim #type: ignore

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

logging.basicConfig(filename="dqn_train_50k.log",
                    format='',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# TODO: make arguments to pass into main
EPSILON_START = 0.9
EPSILON_END = 0.025
NUM_GAMES = 50_000
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / NUM_GAMES
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_MEMORY_SIZE = 10_000
START_REPLAY_MEMORY_SIZE = 5_000
STEPS_FOR_UPDATE = 5_000


# Select to use CPU, CUDA, or MPS
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# Define the transition tuple to store state transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)
    
    
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)

        self.model_path = os.path.join(os.getcwd(),'model.pth')
        print(f'Model path: {self.model_path}')
        print(f"Model found: {os.path.exists(self.model_path)}")

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

        self.env = env
        self.eps = EPSILON_START

        # tracking the last 30 runs
        self.last_thirty = deque(maxlen=30)
            
        if args.test_dqn:
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device))
            self.policy_net.eval()
            print('loading trained model')

        # logging.info(f"Beginning training: {NUM_GAMES} games | decay = {EPSILON_DECAY} | epsilon range {EPSILON_START} - {EPSILON_END}")
        logging.info("episode, epsilon, rolling average (30 games), average loss, average Q values")
    

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device) 
        ## format tensor(BATCH_SIZE, channels, height, width)
        
        if random.random() < self.eps and not test:  # Exploration
            return self.env.get_random_action()
        else:  # Exploitation
            with torch.no_grad():
                q_values = self.policy_net(observation)
                action = q_values.max(1)[1].item()  # Get the index (action) with the highest Q-value
        # implementation 
        return action
    
    # def push(self):
    #     """ You can add additional arguments as you need. 
    #     Push new data to buffer and remove the old one if the buffer is full.
        
    #     Hints:
    #     -----
    #         you can consider deque(maxlen = 10000) list
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
        
    #     ###########################
        
        
    # def replay_buffer(self):
    #     """ You can add additional arguments as you need.
    #     Select batch from buffer.
    #     """
    #     ###########################
    #     # YOUR IMPLEMENTATION HERE #
        
        
        
    #     ###########################
    #     return 
        

    def train(self):
        """
        Implement your training algorithm here
        """
        global_steps = 0

        for episode in range(NUM_GAMES):
           
            episode_steps = 0
            episode_reward = 0
            
            total_loss = 0
            total_q_value = 0
            loss_count = 0

            observation = self.env.reset()  
            
            while True:
                action = self.make_action(observation, test=False)
                next_observation, reward, done, truncated, info = self.env.step(action)
                
                if done or truncated: break

                self.replay_buffer.push(observation, action, next_observation, reward)
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                global_steps += 1
                
                if len(self.replay_buffer) >= START_REPLAY_MEMORY_SIZE and len(self.replay_buffer) >= BATCH_SIZE:
                    step_loss, step_q_value = self.update_policy()
                    total_loss += step_loss
                    total_q_value += step_q_value
                    loss_count += 1
                    
                if global_steps % STEPS_FOR_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())  
                    global_steps = 0
                    
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            avg_q_val = total_q_value / loss_count if loss_count > 0 else 0

            self.last_thirty.append(episode_reward)
            rolling_avg = np.mean(self.last_thirty)
                
            logging.info(f"{episode}, {self.eps:.4f}, {rolling_avg:.4f}, {avg_loss:.4f}, {avg_q_val:.4f}")

            if rolling_avg > 0:
                torch.save(self.policy_net.state_dict(), f'model_50k.pth') # overwrite so latest always exists
                
            self.eps = max(EPSILON_END, self.eps - EPSILON_DECAY)

    def update_policy(self):

        transitions = self.replay_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        # Convert batch data to numpy arrays before converting to tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        state_batch = state_batch.permute(0, 3, 1, 2)  # Permute to (BATCH_SIZE, channels, height, width)

        action_batch = torch.tensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(device).unsqueeze(1)

        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
        next_state_batch = next_state_batch.permute(0, 3, 1, 2)  # Permute to (BATCH_SIZE, channels, height, width)

        # Compute Q-values for current state
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q-values for next state (target)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            
        expected_q_values = reward_batch + (GAMMA * next_q_values)

        # Compute loss (Huber loss)
        huber_loss = F.smooth_l1_loss(q_values, expected_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        huber_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
            
        return (huber_loss.item(), q_values.mean().item())