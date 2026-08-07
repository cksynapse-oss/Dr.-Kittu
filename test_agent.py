import numpy as np
from pymdp import utils
from pymdp.agent import Agent

n_states = [4]
n_actions = [5]
n_obs = [3, 3, 4]

A = utils.obj_array(len(n_obs))
A[0] = np.random.rand(3, 4); A[0] /= A[0].sum(axis=0)
A[1] = np.random.rand(3, 4); A[1] /= A[1].sum(axis=0)
A[2] = np.random.rand(4, 4); A[2] /= A[2].sum(axis=0)

B = utils.obj_array(1)
B[0] = np.random.rand(4, 4, 5); B[0] /= B[0].sum(axis=0)

C = utils.obj_array(len(n_obs))
C[0] = np.random.rand(3)
C[1] = np.random.rand(3)
C[2] = np.random.rand(4)

agent = Agent(A=A, B=B, C=C)
obs = [0, 1, 2]
qs = agent.infer_states(obs)
print("qs", qs)
agent.infer_policies()
action = agent.sample_action()
print("action", action)
