import numpy as np
from pymdp import utils, maths
from pymdp.control import construct_policies, update_posterior_policies

n_states = 4
n_actions = 5
n_obs = [3, 3, 4]

A = utils.obj_array(len(n_obs))
A[0] = np.random.rand(3, 4); A[0] /= A[0].sum(axis=0)
A[1] = np.random.rand(3, 4); A[1] /= A[1].sum(axis=0)
A[2] = np.random.rand(4, 4); A[2] /= A[2].sum(axis=0)

B = utils.obj_array(1)
B[0] = np.random.rand(4, 4, 5); B[0] /= B[0].sum(axis=1, keepdims=True)
# Wait, B is obj_array(num_factors). Each factor has shape (n_states, n_states, n_actions) in 0.0.7!
print("Wait B shape")

