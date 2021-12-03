from .basilisk import basilisk_lm


# Behaviors
def actionDecoder(params, step, history, prev_state) -> dict:
    t = len(history) - 1
    # print(t)
    action_list = params['action_list']
    action_dict = params['action_dict']
    if action_list[t] not in action_dict:
        return {}
    policy_input = action_dict[action_list[t]]
    # We translate path dependent actions like market arb here
    '''
    if policy_input['action_id'] == 'ArbMarket':
        prev_market = prev_state['external']['market']
        amm = params['cfmm']
        trades = amm.trade_to_prices(prev_market, prev_state['AMM'])
        policy_input['arb_trades'] = trades
        print(trades)
    '''
    return policy_input


# Mechanisms
def mechanismHub_state(params, substep, state_history, prev_state, policy_input):
    if 'action_id' not in policy_input:
        return ('state', prev_state['state'])
    # amm = amm_selector.get_amm(params['cfmm_type'])
    action = policy_input['action_id']
    user_id = policy_input['user_id']
    if action == 'Deposit':
        farm_id = policy_input['farm_id']
        asset_pair = policy_input['asset_pair']
        amount = policy_input['amount']
        positions, GlobalPoolData, LiquidityPoolData = basilisk_lm.deposit_shares(prev_state['state'], len(state_history),
                                                                                  user_id, farm_id, asset_pair, amount)
    elif action == 'Claim':
        position = prev_state['state']['positions'][user_id]
        positions, GlobalPoolData, LiquidityPoolData = basilisk_lm.claim_rewards(prev_state['state'], len(state_history), position)
    elif action == 'Withdraw':
        position = prev_state['state']['positions'][user_id]
        positions, GlobalPoolData, LiquidityPoolData = basilisk_lm.withdraw_shares(prev_state['state'], len(state_history), position)
    else:
        return ('state', prev_state['state'])
    return ('state', {'positions': positions, 'GlobalPoolData': GlobalPoolData, 'LiquidityPoolData': LiquidityPoolData})
