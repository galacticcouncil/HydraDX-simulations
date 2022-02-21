import pandas as pd

# Asset balances in the Omnipool:
# Asset 1: 7,000
# Asset 2: 90,000
# Asset 3: 13,000,000
# Asset 4: 8,300,000
# LERNA balances in the Omnipool:
# Against Asset 1: 117,000,000
# Against Asset 2: 109,000,000
# Against Asset 3: 80,000,000
# Against Asset 4: 15,500,000

def initialize_model(initial_lerna_in_pool, initial_tradevolume, initial_fee_assets, initial_fee_HDX, initial_prices, initial_assets_in_pool):

### calculation of LERNA in pool

#initial_liquidity_LERNA = initial_liquidity * prices

    
########## AGENT CONFIGURATION ##########
# key -> token name, value -> token amount owned by agent
# note that token name of 'omniABC' is used for omnipool LP shares of token 'ABC'
# omniHDXABC is HDX shares dedicated to pool of token ABC

    trader = {'HDX': 1000000, 'R1': 1000000, 'R2': 1000000, 'R3': 1000000, 'R4': 1000000, 'R5': 1000000}
    LP1 = {'omniR1': initial_lerna_in_pool[0]}
    LP2 = {'omniR2': initial_lerna_in_pool[1]}
    LP3 = {'omniR3': initial_lerna_in_pool[2]}
    LP4 = {'omniR4': initial_lerna_in_pool[3]}
    LP5 = {'omniR5': initial_lerna_in_pool[4]}

# key -> agent_id, value -> agent dict
    agent_d = {'Trader': trader, 'LP1': LP1, 'LP2': LP2, 'LP3': LP3, 'LP4': LP4, 'LP5': LP5}
    
    
########## MARKET PRESSURE FACTOR #########

## insert here calc + remove hard coded multipliers in action_dict

## to remove market pressure the factor for sell_a1_for_a2 should be the ratio of their initial prices
## factor = price of asset 1 / price of asset 2
## ==>
## sell_r2_r1_factor = initial_prices[0] / initial_prices[1]
## tbd: verify if this direction is correct !
    sell_r2_r1_factor = 1
    sell_r1_r2_factor = 1
    sell_r3_r4_factor = 1
    sell_r4_r3_factor = 1
    
    #sell_r2_r1_factor = initial_prices[0] / initial_prices[1]
    #sell_r1_r2_factor = initial_prices[1] / initial_prices[0]
    #sell_r3_r4_factor = initial_prices[3] / initial_prices[2]
    #sell_r4_r3_factor = initial_prices[2] / initial_prices[3]


###########################################

########## TRADE VOLUME FOR EACH ASSET PAIR ALIGNMENT
## tbd
## trade volumes for each asset should mirror market volumes for the above pool asset values
## insert here calc + adapt action_dict either by factor or array_of_tradevolumes


###########################################

########## ACTION CONFIGURATION ##########

    action_dict = {
        'sell_r2_for_r1': {'token_buy': 'R1', 'token_sell': 'R2', 'amount_sell': sell_r2_r1_factor * initial_tradevolume, 'action_id': 'Trade',
                           'agent_id': 'Trader'},
        'sell_r1_for_r2': {'token_sell': 'R1', 'token_buy': 'R2', 'amount_sell': sell_r1_r2_factor * initial_tradevolume, 'action_id': 'Trade',
                           'agent_id': 'Trader'},
        'sell_r4_for_r3': {'token_buy': 'R3', 'token_sell': 'R4', 'amount_sell': sell_r4_r3_factor * initial_tradevolume, 'action_id': 'Trade',
                           'agent_id': 'Trader'},
        'sell_r3_for_r4': {'token_sell': 'R3', 'token_buy': 'R4', 'amount_sell': sell_r3_r4_factor * initial_tradevolume, 'action_id': 'Trade',
                           'agent_id': 'Trader'}
    }

# list of (action, number of repetitions of action), timesteps = sum of repititions of all actions
    trade_count = 1000
    action_ls = [('trade', trade_count)]

# maps action_id to action dict, with some probability to enable randomness
    prob_dict = {
        'trade': {'sell_r2_for_r1': 0.5,
                  'sell_r1_for_r2': 0,
                  'sell_r4_for_r3': 0.25,
                  'sell_r3_for_r4': 0.25}
    }

########## CFMM INITIALIZATION ##########

    initial_values = {
        'token_list': ['R1', 'R2', 'R3', 'R4', 'R5'],
        #'R': [initial_liquidity[0], initial_liquidity[1], initial_liquidity[2], initial_liquidity[3], initial_liquidity[4]],
        'R': [initial_assets_in_pool[0], initial_assets_in_pool[1], initial_assets_in_pool[2], initial_assets_in_pool[3], initial_assets_in_pool[4]],
        #'P': [2, 2 / 3, 1, 3, 4],
        'P': [initial_prices[0], initial_prices[1], initial_prices[2], initial_prices[3], initial_prices[4]],
        #initial_prices
        'fee_assets': initial_fee_assets,
        'fee_HDX': initial_fee_HDX
    }

############################################ SETUP ##########################################################

    config_params = {
        'cfmm_type': "",
        'initial_values': initial_values,
        'agent_d': agent_d,
        'action_ls': action_ls,
        'prob_dict': prob_dict,
        'action_dict': action_dict,
    }
    return config_params
    #return ('config_params', config_params)
    #return trader, LP1, LP2, agent_d, action_dict, trade_count, action_ls, prob_dict, initial_values, config_params
