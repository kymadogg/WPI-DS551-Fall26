### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
# Modified By Yanhua Li on 09/09/2022 for gym==0.25.2
# Modified By Yanhua Li on 08/19/2023 for gymnasium==0.29.0
# Updated for DS551/CS551 Fall 2026 with gymnasium==1.2.2
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gymnasium.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is the value of state s
    """

    # Please note that reward can be defined on (state), (state, action), (state, action, next_state). In this assignment, we define the reward on (state,action,next_state).

    '''
    Kylie's Notes:

    - nested for loop with states being the outer loop (for nS) and actions being the inner loop (for nA)
    - basically going through every state & action pair and looking at components of P
    - The value of a state V(s) equals the immediate reward plus the discounted value of the next state
    
    - remember that Sigma is really just a for loop
    '''
    
    value_function = np.zeros(nS) 

    d = tol + 1 # d = delta, begin with a higher value than tol to start loop
    while d > tol:
        d = 0
        for state in range(nS): # for the number of states
            state_value = value_function[state] # get state value
            new_value = 0

            for action in range(nA): # at this point is a nested loop iterating through every action for every state available
                for probability, nextstate, reward, terminal in P[state][action]: 
                    # probability = the probability of transitioning from "state" to "nextstate" with "action"
                    # nextstate = literally the index of the next state based on the action taken (i.e. if state = 4 and next move is to go to the right (+1), next state is 5)
                    # reward = either 0 or 1, the reward for transitioning from "state" to "nextstate" with "action"
                    # terminal = True when "nextstate" is a terminal state (hole or goal), False otherwise

                    if not terminal:
                        new_value += policy[state][action] * probability * (reward + value_function[nextstate] * gamma)
                    else: 
                        new_value += policy[state][action] * probability * reward

            value_function[state] = new_value # overwrite state with new calculated value
            diff = abs(state_value - new_value)
            if d < diff: d = diff # update delta if calculated difference is larger

    return value_function 


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.

    TL;DR: should be one pass over all of the states available, updating the actions
    """

    new_policy = np.ones([nS, nA]) / nA # policy as a uniform distribution
    # this really is not used since it gets overwritten later on.

    for state in range(nS): # for each state available
        actions = np.zeros(nA) # make an array to store each action (eventually a nested list)
        for action in range(nA): # for each action in action space (nA)...
            action_value = 0
            for probabilty, nextstate, reward, _ in P[state][action]: # s′,r∑​p(s′,r∣s,a)[r+γV(s′)] iterate through
                action_value += probabilty * (reward + gamma * value_from_policy[nextstate])
            actions[action] = action_value # Q(s,a), which is the value of taking action a in state s
        best_action = np.argmax(actions) # return the index of the highest action available in actions
        new_actions = np.zeros(nA) # new array
        new_actions[best_action] = 1 # π(s)←argamax​Q(s,a), implements the policy improvement step, new_actions[3] = 1 --> [0,0,0,1]
        new_policy[state] = new_actions # π(s) ← best action. inserts a list of new actions into the state index for the new policy
        # print(f"State: {state}, Actions: {actions}, Best Action: {best_action}")

    return new_policy

def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()

    pol = True
    while pol:
        V = policy_evaluation(P, nS, nA, new_policy, gamma, tol)
        update = policy_improvement(P, nS, nA, V, gamma)

        if np.array_equal(update, new_policy):
            pol = False # breaks the loop

        new_policy = update
  
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    # basically identical to policy eval except pi(s) is replaced with 'a' in the V(s) ='s step

    V_new = V.copy()
    policy_new = np.zeros([nS, nA]) # state * action array for the policy (frozen lake is 16 x 4)

    d = tol + 1
    while d > tol: # while delta is greater than the tolerance
        d = 0
        for state in range(nS): # loops through each state
            v = V_new[state]
            action_values = np.zeros(nA)
            for action in range(nA): # loops through actions that can be taken at each state (moving up, down, right, left)
                for probability, nextstate, reward, terminal in P[state][action]:
                    if terminal: 
                        action_values[action] += probability * reward # if terminal is true, the 
                    else:
                        action_values[action] += probability * (reward + gamma * V_new[nextstate])
            V_new[state] = np.max(action_values)
            d = max(d, abs(v - V_new[state])) # get the higher value and set it equal to delta

    policy_new = policy_improvement(P, nS, nA, V_new, gamma)
    return policy_new, V_new


def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game environment from gymnasium, play multiple episodes of the game.
    An episode is over when the returned value for "terminated" or "truncated" is True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have observation_space, action_space, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob, _ = env.reset() # initialize the episode
        done = False
        while not done: # using "not truncated" as well, when using time_limited wrapper.
            if render:
                env.render() # render the game

            action = np.argmax(policy[ob])

            new_ob, rw, term, trun, _ = env.step(action)

            total_rewards += rw
            ob = new_ob
            done = term or trun
            
    return total_rewards


