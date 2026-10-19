#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
# Updated for DS551/CS551 Fall 2026 with gymnasium==1.2.2
import gymnasium as gym
import numpy as np
import random
import sys
from collections import defaultdict
from collections import Counter

from td import *
"""
    This file includes unit test for td.py
    You could test the correctness of your code by
    typing 'python -m nose -v td_test.py' in the terminal
"""

env = gym.make('CliffWalking-v1')
#---------------------------------------------------------------


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def test_python_version():
    '''------Temporal Difference(50 points in total)------'''

    assert sys.version_info[0] == 3  # require python 3

#---------------------------------------------------------------


def test_epsilon_greedy():
    '''epsilon_greedy (0 point)'''
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Q[5][1] = -1
    Q[5][2] = -1
    Q[5][3] = -1
    state = 5
    seed_all(202604)

    actions = []
    for _ in range(10000):
        action = epsilon_greedy(Q, state, 4, epsilon=0.1)
        actions.append(action)

    assert np.allclose(1 - np.count_nonzero(actions) / 10000, 0.925, atol=0.02)

#---------------------------------------------------------------


def test_sarsa():
    '''SARSA (25 points)'''
    seed_all(202605)
    test_policy = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]])
    # note: we do not have cliff state in Q_s if using Sarsa
    Q_s = sarsa(env, n_episodes=50000, gamma=1.0, alpha=0.01, epsilon=0.1)
    print("Q[8]  =", Q_s[8])
    print("Q[9]  =", Q_s[9])
    print("Q[10] =", Q_s[10])
    print("Q[11] =", Q_s[11])
    policy_q = np.array([np.argmax(Q_s[key]) if key in Q_s else -1 for key
                         in np.arange(48)]).reshape((4, 12))
    print(f"POLICY: {policy_q}")
    assert np.allclose(policy_q.shape, (4, 12))
    assert np.allclose(policy_q[2:, ], test_policy)

#---------------------------------------------------------------


def test_q_learning():
    '''Q_learning (25 points)'''
    seed_all(202606)
    Q_q = q_learning(env, n_episodes=10000, gamma=1.0, alpha=0.01, epsilon=0.1)
    policy_q = np.array([np.argmax(Q_q[key]) if key in Q_q else -1 for key
                         in np.arange(48)]).reshape((4, 12))
    test_policy = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]])

    # print(policy_q)
    assert np.allclose(policy_q.shape, (4, 12))
    assert np.allclose(policy_q[2:, ], test_policy)
