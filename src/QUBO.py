# https://docs.pasqal.com/pulser/tutorials/qubo/
import matplotlib.pyplot as plt
import numpy as np
import pulser
import pulser_simulation
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# given Q matrix (always symmetric)
Q = np.array(
    [
        [-10.0, 19.7365809, 19.7365809, 5.42015853, 5.42015853],
        [19.7365809, -10.0, 20.67626392, 0.17675796, 0.85604541],
        [19.7365809, 20.67626392, -10.0, 0.85604541, 0.17675796],
        [5.42015853, 0.17675796, 0.85604541, -10.0, 0.32306662],
        [5.42015853, 0.85604541, 0.17675796, 0.32306662, -10.0],
    ]
)

# Useful because:
# .1 All the off-diagonal terms are positive.
#       The Rydberg interaction is strictly positive, which is why it matters that our off-diagonal terms are too.
# 2. The diagonal terms are all equal.
#       The diagonal terms of Q Q are understood as a detuning to apply on each atom. Since the diagonal terms are all the same, we can use the same detuning δ for all the atoms. If that was not the case, some kind of local addressability capabilities would be necessary. See https://docs.pasqal.com/pulser/tutorials/mwis/

# because the Q matrix is small, we can classically compute optimal solution for reference

bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
costs = []
# this takes exponential time with the dimension of the QUBO
for b in bitstrings:
    z = np.array(list(b), dtype=int)
    cost = z.T @ Q @ z
    costs.append(cost)
zipped = zip(bitstrings, costs)
sort_zipped = sorted(zipped, key=lambda x: x[1])
print(sort_zipped[:3])

# analog device with longer pulse capabilities
device = pulser.DigitalAnalogDevice
device.print_specs()
