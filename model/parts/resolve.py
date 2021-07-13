import numpy as np
import pandas as pd

def s_resolve_H(params, substep, state_history, prev_state, policy_input):
    """
    Resolves and returns H.
    """
    Q = prev_state['Q']
    # place external changes to H

    return ('H', Q)

def s_base_weight(params, substep, state_history, prev_state, policy_input):
    """
    Resolves and returns Wq as Sq.
    """
    ############ is this still valid with new weight to share conversion #################
    ############ current version seems yes, if altering 1 to 1 relationship, then no #####
    Sq = prev_state['Sq']
    # Constraint that Wq is defined as Sq

    return ('Wq', Sq)

# JS July 8, 2021: this method is not used in the V2 Spec
#########
# def s_pool_weight(params, substep, state_history, prev_state, policy_input):
#    """
#    Resolves the weights and returns the 'pool' variable.
#    """
#    pool = prev_state['pool']
#
#    pool.update_weight()
#    return ('pool', pool)
#########