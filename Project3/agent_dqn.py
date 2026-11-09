#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import logging
import os
from argument import add_arguments
import argparse
import torch #type: ignore
import torch.nn.functional as F #type: ignore
import torch.optim as optim #type: ignore
from agent_hyperparams import Params
from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

# maybe this should be removed?
torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

hyp = Params()
# logging.basicConfig(filename="dqn_train_50k.log",
#                     format='',
#                     filemode='w')

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)


# TODO: make arguments to pass into main

# hyper params (hardcode incase import/ arg errors are made by accident)


# hardware selection
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# transition 4 tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            parameters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN,self).__init__(env)

        # arguments for trainung
        self.trials = args.trials
        self.decay = (hyp.eps_start - hyp.eps_end) / self.trials
        self.filename = args.name

        logging.basicConfig(filename=f"dqn_train_{args.name}.log",
                    format='',
                    filemode='w')

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        self.model_path = os.path.join(os.getcwd(),f'model_{args.name}.pth')
        print(f'Model path: {self.model_path}')
        print(f"Model found: {os.path.exists(self.model_path)}")

        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=hyp.learn_rate , amsgrad=True)
        self.memory = deque([], maxlen=hyp.replay_mem)

        self.env = env
        self.eps = hyp.eps_start

        # tracking the last 30 runs
        self.last_thirty = deque(maxlen=30)
            
        if args.test_dqn:
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=device))
            self.policy_net.eval()
            print('loading trained model')
        logging.info("DATA FOR PLOTTING LATER")
        logging.info("episode, epsilon, episode reward, rolling average (30 games), average loss, average Q values")

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
        # format tensor(hyp.batch_sz, channels, height, width)
        
        # explore and exploit
        if random.random() < self.eps and not test:
            return self.env.get_random_action() # exploration: random action
        else:  
            with torch.no_grad():
                q_values = self.policy_net(observation)
                action = q_values.max(1)[1].item()  # exploitation: get action with highest q value
        return action
    
    def push(self, state, action, next_state, reward):
        """
        Push a transition into the replay buffer.

        Old transitions are automatically removed when the
        buffer reaches its maximum capacity.
        """
        self.memory.append(
            Transition(state, action, next_state, reward)
        )
        
        
    def replay_buffer(self):
        """
        Randomly sample a batch of transitions from memory.

        Returns:
            transitions: List of sampled transitions.
        """
        return random.sample(self.memory, hyp.batch_sz) 
        

    def train(self):
        """
        Implement your training algorithm here
        """
        global_steps = 0

        for episode in range(self.trials): # loop through episodes
           
            episode_steps = 0
            episode_reward = 0
            
            total_loss = 0
            total_q_value = 0
            loss_count = 0

            obs = self.env.reset()  
            
            while True: # while episode is still alive 
                action = self.make_action(obs, test=False) # select action with eps greedy
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                if done or truncated: break # end episode if done or trunicated

                self.push(obs, action, next_obs, reward) # store transition in memeory

                # move to next state and update stats
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                global_steps += 1

                # dont start learning until memory has enough episodes (maybe should have made this lower...)
                if len(self.memory) >= hyp.st_replay_mem  and len(self.memory) >= hyp.batch_sz:
                    step_loss, step_q_value = self.update_policy()
                    total_loss += step_loss
                    total_q_value += step_q_value
                    loss_count += 1

                # sync target network with policy network
                if global_steps % hyp.update_steps == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())  
                    global_steps = 0

            # metric calculations
            avg_loss = total_loss / loss_count if loss_count > 0 else 0
            avg_q_val = total_q_value / loss_count if loss_count > 0 else 0

            self.last_thirty.append(episode_reward)
            rolling_avg = np.mean(self.last_thirty)

            # logging for later plotting
            logging.info(f"{episode}, {self.eps:.4f}, {episode_reward:.4f}, {rolling_avg:.4f}, {avg_loss:.4f}, {avg_q_val:.4f}")

            # save model if above a rolling average threshold
            if rolling_avg > 0:
                torch.save(self.policy_net.state_dict(), f'model_{self.filename}.pth') # overwrite so latest always exists

            if self.trials / 2 == episode: # halfway point for comparaison
                torch.save(self.policy_net.state_dict(), f'model_{self.filename}_halfway.pth') # overwrite so latest always exists
            
            # increment episilon
            self.eps = max(hyp.eps_end, self.eps - self.decay)

    def update_policy(self):
        """
        Update the policy network using a random batch from replay memory.

        Samples transitions, computes current and target Q-values, calculates
        the Huber loss, and performs one optimization step with gradient clipping.

        Returns:
            tuple[float, float]:
                The loss value and mean predicted Q-value for the sampled batch.
        """

        transitions = self.replay_buffer() # returns a random sample from memory
        batch = Transition(*zip(*transitions))
        
        # convert batch --> numpy --> tensor
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        state_batch = state_batch.permute(0, 3, 1, 2)  # pm to (hyp.batch_sz, channels, height, width)

        action_batch = torch.tensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32).to(device).unsqueeze(1)

        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
        next_state_batch = next_state_batch.permute(0, 3, 1, 2)  # pm to (hyp.batch_sz, channels, height, width)

        # Compute Q-values for current state
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q-values for next state (target)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            
        expected_q_values = reward_batch + (hyp.gamma * next_q_values)

        # huber loss
        huber_loss = F.smooth_l1_loss(q_values, expected_q_values)

        # policy network optimization
        self.optimizer.zero_grad()
        huber_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
            
        return (huber_loss.item(), q_values.mean().item())