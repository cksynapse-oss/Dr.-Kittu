import numpy as np
from pymdp import utils, maths
class HabitCoachAgent:
    """
    Active inference agent for personal habit & mood coaching.
    Operates on discrete states and observations. Supports online learning.
    """

    def __init__(self, lr_A=0.1, lr_B=0.1):
        self.lr_A = lr_A
        self.lr_B = lr_B

        # ---- States ----
        self.state_names = ["focused", "stressed", "tired", "calm"]
        self.n_states = len(self.state_names)

        # ---- Observations ----
        self.obs_screen = ["low", "med", "high"]
        self.obs_steps  = ["low", "med", "high"]
        self.obs_mood   = ["good", "neutral", "bad", "none"]
        self.n_screen = len(self.obs_screen)
        self.n_steps = len(self.obs_steps)
        self.n_mood = len(self.obs_mood)
        self.n_obs = [self.n_screen, self.n_steps, self.n_mood]

        # ---- Actions ----
        self.action_names = ["do_nothing", "breathe", "walk", "soundscape", "social_nudge"]
        self.n_actions = len(self.action_names)

        # ---- Initialize A matrices (observation likelihoods) ----
        # A[modality][state, observation]
        self.A = utils.obj_array(len(self.n_obs))
        self.A[0] = np.array([  # screen
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.1, 0.3, 0.6],
            [0.7, 0.2, 0.1]
        ])
        self.A[1] = np.array([  # steps
            [0.1, 0.8, 0.1],
            [0.6, 0.3, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.3, 0.6]
        ])
        self.A[2] = np.array([  # mood
            [0.7, 0.2, 0.1, 0.0],
            [0.1, 0.3, 0.6, 0.0],
            [0.1, 0.5, 0.4, 0.0],
            [0.8, 0.1, 0.1, 0.0]
        ])

        # Prior counts for Dirichlet (for learning)
        self.A_counts = utils.obj_array(len(self.n_obs))
        for m in range(len(self.n_obs)):
            self.A_counts[m] = np.ones_like(self.A[m]) * 1.0  # weak uniform prior

        # ---- Initialize B matrices (state transitions) ----
        # B[action][state_from, state_to]
        self.B = utils.obj_array(self.n_actions)
        # Base persistence for do_nothing
        self.B[0] = np.eye(self.n_states) * 0.8 + 0.05
        # Hand-crafted transitions (will be learned)
        self.B[1] = np.array([  # breathe
            [0.8, 0.1, 0.0, 0.1],
            [0.1, 0.2, 0.1, 0.6],
            [0.1, 0.1, 0.6, 0.2],
            [0.1, 0.1, 0.1, 0.7]
        ])
        self.B[2] = np.array([  # walk
            [0.7, 0.2, 0.3, 0.2],
            [0.1, 0.3, 0.1, 0.2],
            [0.1, 0.3, 0.5, 0.3],
            [0.1, 0.2, 0.1, 0.3]
        ]).T
        self.B[3] = np.array([  # soundscape
            [0.8, 0.1, 0.0, 0.1],
            [0.1, 0.3, 0.1, 0.5],
            [0.1, 0.2, 0.6, 0.1],
            [0.1, 0.3, 0.1, 0.5]
        ])
        self.B[4] = np.array([  # social_nudge
            [0.7, 0.2, 0.1, 0.2],
            [0.1, 0.3, 0.2, 0.1],
            [0.1, 0.4, 0.4, 0.1],
            [0.1, 0.1, 0.3, 0.6]
        ])

        # Prior counts for B
        self.B_counts = utils.obj_array(self.n_actions)
        for act in range(self.n_actions):
            self.B[act] = self.B[act] / self.B[act].sum(axis=1, keepdims=True)
            self.B_counts[act] = np.ones_like(self.B[act]) * 1.0

        # ---- Preferences (C) ----
        self.C = utils.obj_array(len(self.n_obs))
        self.C[0] = np.array([0.1, 0.8, 0.1])     # screen: prefer medium
        self.C[1] = np.array([0.1, 0.3, 0.6])     # steps: prefer high
        self.C[2] = np.array([0.7, 0.2, 0.05, 0.05])  # mood: prefer good > neutral

        # ---- Initial state prior (D) ----
        self.D = np.array([0.25, 0.25, 0.25, 0.25])

        # ---- Internal belief ----
        self.qs = self.D.copy()  # current posterior over states

    def perceive(self, obs_dict):
        """
        Update belief given a dictionary of observations.
        obs_dict keys: 'screen', 'steps', 'mood'. Values are int indices, or None if missing.
        """
        log_likelihood = np.zeros(self.n_states)
        modality_present = False

        # Screen observation
        if obs_dict.get('screen') is not None:
            log_likelihood += np.log(self.A[0][:, obs_dict['screen']] + 1e-16)
            modality_present = True

        # Steps observation
        if obs_dict.get('steps') is not None:
            log_likelihood += np.log(self.A[1][:, obs_dict['steps']] + 1e-16)
            modality_present = True

        # Mood observation
        if obs_dict.get('mood') is not None:
            log_likelihood += np.log(self.A[2][:, obs_dict['mood']] + 1e-16)
            modality_present = True

        if modality_present:
            self.qs = maths.softmax(np.log(self.qs + 1e-16) + log_likelihood)

    def plan(self):
        """
        Compute expected free energy for each one-step policy and return the best action index.
        Uses the current belief (self.qs).
        """
        G = np.zeros(self.n_actions)
        for u in range(self.n_actions):
            # Predictive state distribution given action u
            # B[u] has shape (n_states, n_states) where B[u][s_from, s_to] = P(s_to | s_from, u)
            qs_prime = self.B[u].T.dot(self.qs)
            
            efe = 0.0
            for m in range(len(self.n_obs)):
                # Predictive observation distribution
                # A[m] has shape (n_states, n_obs)
                pred_obs = qs_prime.dot(self.A[m])
                
                # Expected Utility
                eu = pred_obs.dot(np.log(self.C[m] + 1e-16))
                
                # Expected Ambiguity
                H_m = -np.sum(self.A[m] * np.log(self.A[m] + 1e-16), axis=1)
                ambiguity = qs_prime.dot(H_m)
                
                # EFE = Ambiguity - Expected Utility
                efe += ambiguity - eu
                
            G[u] = efe
            
        best_action = np.argmin(G)
        return best_action

    def act(self, obs_dict):
        """
        Full perception + action selection. Returns action index.
        """
        self.perceive(obs_dict)
        return self.plan()

    def learn(self, obs_dict, prev_qs, prev_action):
        """
        Online Dirichlet updates for A and B using the last transition.
        prev_qs: belief before the action (numpy array)
        prev_action: action taken (int)
        obs_dict: current observations (dict, same format as perceive)
        """
        # ---- Update A (observation model) ----
        if obs_dict.get('screen') is not None:
            self.A_counts[0][:, obs_dict['screen']] += self.lr_A * self.qs
        if obs_dict.get('steps') is not None:
            self.A_counts[1][:, obs_dict['steps']] += self.lr_A * self.qs
        if obs_dict.get('mood') is not None:
            self.A_counts[2][:, obs_dict['mood']] += self.lr_A * self.qs

        # Re-normalise A from counts
        for m in range(len(self.n_obs)):
            self.A[m] = self.A_counts[m] / self.A_counts[m].sum(axis=1, keepdims=True)

        # ---- Update B (transition model) ----
        prev_state = np.argmax(prev_qs)
        curr_state = np.argmax(self.qs)
        self.B_counts[prev_action][prev_state, curr_state] += self.lr_B

        # Re-normalise B rows
        for act in range(self.n_actions):
            row_sums = self.B_counts[act].sum(axis=1, keepdims=True)
            self.B[act] = self.B_counts[act] / row_sums

    def save(self, filepath):
        """Save model parameters to a .npz file."""
        np.savez(filepath,
                 A_counts=np.array([ac.tolist() for ac in self.A_counts], dtype=object),
                 B_counts=np.array([bc.tolist() for bc in self.B_counts], dtype=object),
                 D=self.D)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model parameters from a .npz file."""
        data = np.load(filepath, allow_pickle=True)
        self.A_counts = utils.obj_array(len(self.n_obs))
        for m, arr in enumerate(data['A_counts']):
            self.A_counts[m] = arr
            self.A[m] = arr / arr.sum(axis=1, keepdims=True)
        self.B_counts = utils.obj_array(self.n_actions)
        for a, arr in enumerate(data['B_counts']):
            self.B_counts[a] = arr
            self.B[a] = arr / arr.sum(axis=1, keepdims=True)
        self.D = data['D']
        self.qs = self.D.copy()
        print(f"Model loaded from {filepath}")
