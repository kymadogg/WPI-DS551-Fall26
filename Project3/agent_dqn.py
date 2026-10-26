#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import deque, namedtuple
import logging
import os
from argument import add_arguments

import torch #type:ignore
import torch.nn as nn #type:ignore
import torch.optim as optim #type:ignore

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

logger = logging.getLogger(__name__)

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

        # hyper params
        self.epsilon_start = 1
        self.epsilon_end = 0.025
        self.epsilon_decay = 1000        
        self.clip_gradient = 1
        self.tau = 0.001
        self.gamma = 0.99

        # training params
        self.batch_size = 16 
        self.learning_rate = 0.0001
        self.num_episodes = args.trials
        
        # naming
        self.filename = args.name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # logging config
        logging.basicConfig(
            filename=f"dqn_train_{args.name}.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w",
        )
        logger.setLevel(logging.DEBUG)

        # model weights path
        self.model_path = os.path.join(os.getcwd(),f'model_{args.name}.pth')
        print(f'Model path: {self.model_path}')
        print(f"Model found: {os.path.exists(self.model_path)}")

        # nets
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)

        # state disctionaries
        self.policy_net.load_state_dict(torch.load("outputs_final/models_final/model_old.pth", map_location=torch.device(self.device)))
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # docs: 
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        self.transition = namedtuple("Transition", ("state", "action", "next_state", "reward")) # 4 tuple
        self.buffer = deque([], maxlen=10000) # memory

        # initialize counters
        self.episode = 0
        self.action_steps_done = 0
        self.steps = 0

        # tracking the last 30 runs
        self.last_thirty = deque(maxlen=30)

        # for running with --test_dqn
        if args.test_dqn:
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.policy_net.eval()
            print('loading trained model')

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

        actions = [0,1,2,3]

        sample = random.random()

        self.epsilon_threshold = self.epsilon_start - (self.epsilon_start - self.epsilon_end) / np.float32(self.num_episodes) * np.float32(self.episode)
        self.action_steps_done += 1

        if test or sample > self.epsilon_threshold: # explore or exploit
            with torch.no_grad(): # exploitation: pick the largest Q value
                observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
                observation_tensor = observation_tensor.permute([2,0,1]).unsqueeze(0)
                outputs = self.policy_net(observation_tensor)
                action = torch.argmax(outputs).item()
        else: # exploration
            action = random.choice(actions)
        return action
    
    def push(self, *args):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        self.buffer.append(self.transition(*args))
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        return random.sample(self.buffer, self.batch_size)
        
        
    def train(self):
        """
        Implement your training algorithm here
        """
        
        for e in range(self.num_episodes):
            self.episode = e
            state = self.env.reset()
            t_reward = 0.0 # total reward
            t_i = 0 # index
            
            while True:
                action = self.make_action(state)
                observation, reward, done, truncated, _ = self.env.step(action)
                t_reward += reward
                
                next_state = None if done or truncated else observation
                self.push(state, action, next_state, reward)
    
                state = next_state

                if len(self.buffer) < self.batch_size: # could be if len(self.replay_memory) < self.batch_size: maybe?
                    continue
                
                transitions = self.replay_buffer()
                batch = self.transition(*zip(*transitions))
                
                non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
                non_final_next_state_list = [torch.tensor(s, device=self.device).unsqueeze(0) for s in batch.next_state if s is not None]        
                non_final_next_state = torch.cat(non_final_next_state_list, dim=0)
            
                state_batch = torch.tensor(np.array(batch.state), device=self.device)
                action_batch = torch.tensor(batch.action, device=self.device)
                reward_batch = torch.tensor(batch.reward, device=self.device)
                state_batch = state_batch.permute([0,3,1,2])
                
                batch_policy = torch.zeros(self.batch_size, device=self.device)
                batch_policy_all = self.policy_net(state_batch)
                
                for idx in range(self.batch_size):
                    
                    batch_policy[idx] = batch_policy_all[idx][action_batch[idx]]
                    
                batch_target_next = torch.zeros(self.batch_size, device=self.device)

                with torch.no_grad():
                    non_final_next_state = non_final_next_state.permute([0,3,1,2])
                    batch_target_next[non_final_mask] = self.target_net(non_final_next_state).max(1)[0]
            
                expected_state_action_values = ( batch_target_next*self.gamma ) + reward_batch
                criterion = nn.SmoothL1Loss()
                loss = criterion(batch_policy, expected_state_action_values)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.clip_gradient)
                self.optimizer.step()
                                
                self.step_counter += 1
                if self.step_counter % 100 == 0:
                    logger.info(
                        "step=%d epsilon=%.6f",
                        self.step_counter,
                        self.epsilon_threshold
                    )
                                
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    break
                
                t_i += 1 # iterate
            
            print(f"Episode Number: {e} | avg reward: {np.mean(self.last_thirty)}")
            
            self.last_thirty.append(t_reward)
            
            # Calculate and log average reward over the last 30 episodes
            logger.info("episode=%d average_reward_last_30=%.4f", e, np.mean(self.last_thirty))

            if e % 10 == 0:
                logger.info("episode=%d reward=%.4f max_steps=%d", e, t_reward, t_i,)

            if e % 100 == 0:
                model_path = f"model_{e}.pth"
                torch.save(self.policy_net.state_dict(), model_path)