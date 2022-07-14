import itertools


"""
Model parameters can be set here & the model is initialized

"""
############### TRADES ################################################################
#### Passes a trade event to action_list.py that will be executed for each timestep
#### Is only relevant if 'exo_random_sequence' is 'off'
#### This trade event is overruled by events defined by the random sequence
# exo_trade = [True] #['noisy', 'r_skewed', 'q_skewed', 'high_magnitude', 'high_frequency']
# exo_trade = ['noisy', 'r_skewed', 'q_skewed', 'high_magnitude', 'high_frequency']
# exo_trade = ['test_q_for_r'] # agent 0
# exo_trade = ['test_r_for_q'] # agent 1
# exo_trade = ['pass'] # No trades 
exo_trade = ['test_r_for_r'] # agent 1

################## LIQUIDITY ######################################################
#### Passes a liquidity event to action_list.py that will be executed for each timestep
#### Is only relevant if 'exo_random_sequence' is 'off'
#### This liquidity event is overruled by events defined by the random sequence
# exo_liq = ['test_add']  # agent 2
# exo_liq = ['test_remove'] # agent 3
exo_liq = ['pass'] # No liquidity actions 


################## TRADE SIZE ########################################
#### Sets the trade size, regardless of direction or asset type
#### This value is assigned to the variable 'trade_random_size' of the dataframe
#### For sigma = 0 the trade size will always be equal to mu, otherwise it will be drawn from a normal distribution
#### The method 's_trade_random' defined in action_list.py uses these parameters and assigns a trade size to each timestep
# exo_trade_size = ['fix_size']  
exo_trade_size = ['vary_size'] 
mu = [1000]
sigma = [100]

############### RANDOM SEQUENCE - add -swap -remove #####################
#### If 'on' then a random sequence for agent actions is generated in action_list.py
#### This sequence can be defined there in detail to assign particular actions to each timestep
#### Currently there is an add liquidity event at t=10, a remove liquidity event at t=90 and random swaps in between
#### These actions can be changes to accomodate any desired test case and user behaviors
#### Also, if 'on' the same agent who provides liquidity also removes it (currently agent 2)
#### If 'off' the same action is performed for each timestep as defined by above parameter 'exo_trade' and 'exo_liq'
#exo_random_sequence = ['off'] 
exo_random_sequence = ['on'] 
# exo_random_sequence = ['q_to_i_i_q'] 



############## MULTIPLE ASSET ACTION TYPE #########################
#exo_asset = ['alternating']
exo_asset = ['i_only']
# exo_asset = ['j_only']


############## COMPOSITE ACTION TYPE #########################
exo_composite = ['alternating'] # add liq then trade
# exo_composite = ['trade_bias'] # add liq then trade

###################### ASYMMETRIC / SYMMETRIC LIQUIDITY #################################
# Uniswap permits symmetric adds , this will allow a naive implementation of assymetric add and removes, 
# but must account for the correct comparison to another system 
ENABLE_SYMMETRIC_LIQ = [True] 
# ENABLE_SYMMETRIC_LIQ = [False] # False


#################### CHANGE LOG PARAMETER #######################
#### This parameter can be used for future exploration of alternative mechanisms
#### New mechanisms can be tested on the basis of the date of their implementation (linking to the respective specification document)
#### After this new date is introduced here, the mechanisms can be included into the mechanismHubs in v2_hydra.py
CHANGE_LOG = ['7-13-21'] 

#ACTION_LIST = [['test_add', 'test_r_for_q', 'test_q_for_r','test_r_for_r', 'test_remove']]
ACTION_LIST = [['test_add', 'test_r_for_r', 'test_remove']]
asset_initial_values = {
    'i' : 
        {'R': 1000000,
        'Q': 2000000,
        # 'S': 100000000 * 200000000},
        'S': 10000},

    'j' : 
        {'R': 1000000,
        'Q': 2000000,
        # 'S': 100000000 * 200000000},
        'S': 10000},
}

##########  Risk Asset Pool Constant Curvature (cf. V2 Spec)
#### Multiple values will cause a parameter sweep
# a = [0.9, 0.92, 0.94, 0.96, 0.98, 1, 1.02, 1.04, 1.06, 1.08, 1.1]
#a = [0.9, 1, 1.1]
#a = [0.8, 1.2]
a = [1.0]

# V2 Spec June 28th 2021: Initialization of coefficients based upon adding new asset to pool

##################################### a #################################
###### using first element for a
###### needs to be looped over for using a in determining inital state(s)
temp_a = a[0]
#########################################################################

# Assume asset 'i' is added first
Ci = 1
# Next add asset 'j' according to price invariance (cf. V2 Spec)
initial_price_j = asset_initial_values['j']['Q']/asset_initial_values['j']['R']
Cj = ( initial_price_j * (asset_initial_values['j']['R'])**(temp_a+1) ) / ( asset_initial_values['i']['Q']*asset_initial_values['i']['R'] )

# V2 Spec June 28th 2021: Initialization of Risk Asset Pool Constant Y
Y = ( Ci * asset_initial_values['i']['R']**(-temp_a) + Cj * asset_initial_values['j']['R']**(-temp_a) )**(-1/temp_a)

# JS July 8th 2021: This constant is no longer used in V2, and can be removed where it occurs elsewhere in the code
C = asset_initial_values['i']['S'] * asset_initial_values['j']['S']**2 # squared for same as asset k 


ENABLE_BALANCER_PRICING = [True] 

initial_values = {
    'UNI_Qi': asset_initial_values['i']['Q'],
    'UNI_Ri': asset_initial_values['i']['R'],
    'UNI_Si': asset_initial_values['i']['S'],
    'UNI_Qj': asset_initial_values['j']['Q'],
    'UNI_Rj': asset_initial_values['j']['R'],
    'UNI_Sj': asset_initial_values['j']['S'],
    'UNI_ij': asset_initial_values['i']['R'],
    'UNI_ji': asset_initial_values['j']['R'],
    'UNI_Sij': asset_initial_values['j']['R']*asset_initial_values['i']['R'],
    'Ri': asset_initial_values['i']['R'],
    'Ci': Ci,
    # 'Si': 5*asset_initial_values['i']['R'],
    'Si': asset_initial_values['i']['S'],
    'Rj': asset_initial_values['j']['R'],
    'Cj': Cj,
    # 'Sj': 5*asset_initial_values['j']['R'],
    'Sj': asset_initial_values['j']['S'],
    'Sq': asset_initial_values['i']['S'] + asset_initial_values['j']['S'],
    # 'Sq': 5*Q,
    'Q':  asset_initial_values['i']['Q'] + asset_initial_values['j']['Q'],
    'H':  asset_initial_values['i']['Q'] + asset_initial_values['j']['Q'],
    # Hydra initial Y value
    'Y': Y
}
# print(initial_values['Q'])
#################################################################################################################

############################# FEE ############################################
#### These are the parameters of Uniswap that represent the fee collected on each swap. Notice that these are hardcoded in the Uniswap smart contracts, but we model them as parameters in order to be able to do A/B testing and parameter sweeping on them in the future.
fee_numerator = [1000]
fee_denominator = [1000]

### Parameters
factors = [fee_numerator, fee_denominator, exo_trade, exo_liq, ENABLE_SYMMETRIC_LIQ, exo_asset, exo_composite, ACTION_LIST, CHANGE_LOG, a, ENABLE_BALANCER_PRICING]
product = list(itertools.product(*factors))
fee_numerator, fee_denominator, exo_trade, exo_liq, ENABLE_SYMMETRIC_LIQ, exo_asset, exo_composite, ACTION_LIST, CHANGE_LOG, a, ENABLE_BALANCER_PRICING = zip(*product)
fee_numerator = list(fee_numerator)
fee_denominator = list(fee_denominator)
exo_trade =  list(exo_trade)
exo_liq =  list(exo_liq)
ENABLE_SYMMETRIC_LIQ = list(ENABLE_SYMMETRIC_LIQ)
exo_asset =  list(exo_asset)
exo_composite = list(exo_composite)
ACTION_LIST =  list(ACTION_LIST)
CHANGE_LOG = list(CHANGE_LOG)
a = list(a)
ENABLE_BALANCER_PRICING = list(ENABLE_BALANCER_PRICING)


params = {
    'fee_numerator': fee_numerator,
    'fee_denominator': fee_denominator,
    'exo_trade': exo_trade, 
    'exo_liq' : exo_liq,
    'ENABLE_SYMMETRIC_LIQ' : ENABLE_SYMMETRIC_LIQ,
    'exo_asset' : exo_asset,
    'exo_composite' : exo_composite,
    'ACTION_LIST': ACTION_LIST,
    'CHANGE_LOG': CHANGE_LOG,
    'a': a, 
    'ENABLE_BALANCER_PRICING': ENABLE_BALANCER_PRICING,
    'exo_trade_size': exo_trade_size,
    'mu': mu,
    'sigma': sigma,
    'exo_random_sequence': exo_random_sequence,
}

