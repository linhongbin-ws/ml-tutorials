import numpy as np
import sys
import matplotlib.pyplot as plt
# if "../" not in sys.path:
#   sys.path.append("../")

#%%
def value_iteration_for_gamblers(p_h, theta=0.0001, discount_factor=1.0):
    """
    Args:
        p_h: Probability of the coin coming up heads
    """

    # # initialize
    # rewards = np.zeros(101) # 0 to 100
    # rewards[100] = 1 # rewards for s=100 or larger is 1, otherwise is zero
    V = np.zeros(101) # Value for state 1 to 99
    rewards = np.zeros(101)
    rewards[100] = 1
    policy = np.zeros(101) # optimal action for 99 states
    epoch = 0


    def one_step_lookahead(s, V, rewards):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            s: The gambler’s capital. Integer.
            V: The vector that contains values at each state.
            rewards: The reward vector.

        Returns:
            A vector containing the expected value of each action.
            Its length equals to the number of actions.
        """
        #print(min(s, 100-s))
        a = np.arange(0, min(s, 100-s) + 1)

        A = np.zeros(a.shape)
        for i in range(len(A)):
            A[i] = discount_factor*((V[s-a[i]])+rewards[s-a[i]]) *(1-p_h)\
                +  discount_factor*(V[s+a[i]]+rewards[s+a[i]])*p_h

        return A

    while(True):
        Delta = 0
        for s in range(1, 100): # 1 to 99
            A = one_step_lookahead(s, V, rewards)
            best_a_idx = np.argmax(A) # choose the best action
            policy[s] = best_a_idx # how many bet the gambler stake
            v_prime = A[best_a_idx]
            Delta = max(np.abs(v_prime-V[s]), Delta)
            V[s] = v_prime
        epoch+=1
        print("epoch: {0}, Delta: {1}".format(epoch, Delta))
        if Delta < theta:
            print("finish value iteration training")
            break

    return policy, V

#%%
policy, v = value_iteration_for_gamblers(0.25)
print("Optimized Policy:")
print(policy)
print("")

print("Optimized Value Function:")
print(v)
print("")
#%%

