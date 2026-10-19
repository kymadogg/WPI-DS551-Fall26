#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:11:22 2019

@author: huiminren
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
# Updated for DS551/CS551 Fall 2026 with gymnasium==1.2.2
"""
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.

    You could test the correctness of your code
    by typing 'python -m nose -v mc_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and hit otherwise

    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    if observation[0] >= 20:
        return 0
    else:
        return 1


def mc_prediction(policy, env, n_episodes, gamma=1.0):
    """Given policy using sampling to calculate the value function
        by using Monte Carlo first visit algorithm.

    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        Gymnasium environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """

    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)

    for e in range(n_episodes):
        e = []
        s, _ = env.reset() # initialize states
        done = False
        while not done:
            a = policy(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            e.append((s, a, r))
            s = s_next

        G = 0
        for t in reversed(range(len(e))):
            s_t, a_t, reward_t = e[t]
            G = gamma * G + reward_t
            if s_t not in [x[0] for x in e[:t]]:
                returns_sum[s_t] += G
                returns_count[s_t] += 1.0
                V[s_t] = returns_sum[s_t] / returns_count[s_t]
    return V


def epsilon_greedy(Q, state, nA, epsilon=0.1):
    """Selects epsilon-greedy action for supplied state.

    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    state: 
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1

    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 - epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
   
    a_val = Q[state]
    a_best = np.argmax(a_val)
    if random.random() < epsilon:
        a = random.randint(0, nA - 1)
    else:
        a = a_best

    return a


def mc_control_epsilon_greedy(env, n_episodes, gamma=1.0, epsilon=0.1):
    """Monte Carlo control with exploring starts.
        Find an optimal epsilon-greedy policy.

    Parameters:
    -----------
    env: function
        Gymnasium environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-0.1/n_episode during each episode
    and episode must > 0.
    """

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for ep_idx in range(n_episodes):
        # decay epsilon
        epsilon = max(0.1, epsilon - 0.1 / n_episodes)
        
        history = []
        state, _ = env.reset()
        active = True
        
        # generate an episode sequence
        while active:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, term, trunc, _ = env.step(action)
            
            history.append((state, action, reward))
            active = not (term or trunc)
            state = next_state

        # First-visit policy eval through the history
        g_return = 0
        total_steps = len(history)
        
        for t in reversed(range(total_steps)):
            state_t, action_t, reward_t = history[t]
            g_return = gamma * g_return + reward_t
            
            # see if it appears in history
            if (state_t, action_t) not in [(step[0], step[1]) for step in history[:t]]:
                pair = (state_t, action_t)
                returns_sum[pair] += g_return
                returns_count[pair] += 1.0
                Q[state_t][action_t] = returns_sum[pair] / returns_count[pair]

    return Q
