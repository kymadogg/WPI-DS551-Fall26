#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
# Updated for DS551/CS551 Fall 2026 with gymnasium==1.2.2
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Temporal Difference
    In this problem, you will implememnt an AI player for cliffwalking.
    The main goal of this problem is to get familar with temporal diference algorithm.
    You could test the correctness of your code 
    by typing 'python -m nose -v td_test.py' in the terminal.
'''
#-------------------------------------------------------------------------


# from my project2-1 code 
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
   
    action_values = Q[state]
    best_action = np.argmax(action_values)
    if random.random() < epsilon:
        action = random.randint(0, nA - 1)
    else:
        action = best_action

    return action


def sarsa(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    """On-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        Gymnasium environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    """
    
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for _ in range(n_episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q, s, env.action_space.n, epsilon)
        done = False

        while not done:
            next_state, r, term, trunc, _ = env.step(a)
            done = term or trunc

            if done:
                # terminal step
                target = r
                Q[s][a] += alpha * (target - Q[s][a])
            else:
                # not terminal step
                next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
                target = r + gamma * Q[next_state][next_action]
                Q[s][a] += alpha * (target - Q[s][a])
                
                s = next_state
                a = next_action

    return Q


def q_learning(env, n_episodes, gamma=1.0, alpha=0.5, epsilon=0.1):
    '''20 points'''
    """Off-policy TD control. Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        Gymnasium environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor, range between 0 and 1
    alpha: float
        step size, range between 0 and 1
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where A[s][a] is the estimated action value corresponding to state s and action a. 
    """
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # my variable names
    # r = reward
    # a = action
    # s = state
    # tru = trunicated
    # term = terminated

    for _ in range(n_episodes):
        s, _ = env.reset() # no action since episode got reset i think
        done = False # trigger while loop

        while not done:
            a = epsilon_greedy(Q, s, env.action_space.n, epsilon)
            s_next, r, term, tru, _ = env.step(a) # take a step using the action picked by epsilon_greedy()
            done = term or tru # if either are true than the episode is done

            t = r + gamma * np.max(Q[s_next]) # calculate target

            Q[s][a] += alpha * (t - Q[s][a]) 
            s = s_next # iterate by making the next state = to the current one
    return Q
