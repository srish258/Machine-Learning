from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        for i in range(0, S):
            alpha[i][0] = self.pi[i] * self.B[i][self.obs_dict.get(Osequence[0], len(self.obs_dict))]

        for j in range(1, L):
            for i in range(0, S):
                x = 0.0
                for k in range(0, S):
                    x += self.A[k][i] * alpha[k][j - 1]
                alpha[i][j] = self.B[i][self.obs_dict.get(Osequence[j], len(self.obs_dict))] * x
        return alpha
        ######################################################


        

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        for i in range(0, S):
            beta[i][L - 1] = 1

        for j in range(L - 2, -1, -1):
            for i in range(0, S):
                x = 0.0
                for k in range(0, S):
                    x += self.A[i][k] * self.B[k][self.obs_dict.get(Osequence[j + 1], len(self.obs_dict))] * beta[k][
                        j + 1]
                beta[i][j] = x
        return beta
        #######################################################


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        prob = 0
        S = len(self.pi)
        L = len(Osequence)
        alpha = self.forward(Osequence)

        for i in range(0, S):
            prob += alpha[i][L - 1]
        return prob
        #####################################################


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sequence_prob = self.sequence_prob(Osequence)
        for i in range(0, S):
            for j in range(0, L):
                prob[i][j] = (alpha[i][j] * beta[i][j]) / sequence_prob
        return prob
        ######################################################################


    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        sequence_prob = self.sequence_prob(Osequence)

        for i in range(S):
            for j in range(S):
                for k in range(L - 1):
                    index = self.obs_dict.get(Osequence[k + 1])
                    prob[i, j, k] = (alpha[i, k] * beta[j, k + 1] * self.A[i, j] * self.B[j, index]) / sequence_prob
        return prob
        ###################################################
        

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        delta_global = np.zeros([self.pi.shape[0], len(Osequence) - 1], dtype=int)

        # initial prob
        delta_local = self.pi * self.B[:, self.obs_dict.get(Osequence[0], len(self.obs_dict))]

        for l in range(1, len(Osequence)):
            delta_local = (delta_local * self.A.T).T * self.B[:, self.obs_dict.get(Osequence[l], len(self.obs_dict))]
            delta_global[:, l - 1] = np.argmax(delta_local, axis=0)
            delta_local = np.max(delta_local, axis=0)

        path.append(np.argmax(delta_local))
        for i in range(len(Osequence) - 1):
            path.append(delta_global[path[i]][delta_global.shape[1] - 1 - i])

        reverse_states = {v: k for k, v in self.state_dict.items()}
        path = [reverse_states[i] for i in path]

        path = path[::-1]
        ################################################################################
        
        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
