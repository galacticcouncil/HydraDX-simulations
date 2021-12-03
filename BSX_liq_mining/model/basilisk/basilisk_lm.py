

class GlobalPool:
    def __init__(self, id, updated_at, reward_currency, yield_per_period, planned_yielding_periods, blocks_per_period,
                 owner, incentivized_token, max_reward_per_period):
        self.updated_at = updated_at
        self.total_shares = 0
        self.accumulated_rps_start = 0
        self.accumulated_rps = 0
        self.accumulated_rewards = 0
        self.paid_accumulated_rewards = 0
        self.yield_per_period = yield_per_period
        self.blocks_per_period = blocks_per_period
        self.incentivized_token = incentivized_token
        self.reward_currency = reward_currency
        self.max_reward_per_period = max_reward_per_period
        self.planned_yielding_periods = planned_yielding_periods
        self.owner = owner
        self.liq_pools_count = 0
        self.id = id
        self.free_balance = 0


class LiquidityPool:
    def __init__(self, id, updated_at, loyalty_curve, multiplier):
        self.id = id
        self.updated_at = updated_at
        self.total_shares = 0
        self.accumulated_rps = 0
        self.loyalty_curve = loyalty_curve
        self.stake_in_global_pool = 0
        self.multiplier = multiplier
        self.incentivized_token_in_amm = 0
        self.free_balance = 0


class LoyaltyCurve:
    def __init__(self, initial_reward_percentage: float = 0.5, scale_coef: int = 100):
        self.initial_reward_percentage = initial_reward_percentage
        self.scale_coef = scale_coef


def create_farm(params, origin, total_rewards, planned_yielding_periods, blocks_per_period, incentivized_token, reward_currency,
                owner, yield_per_period):
    planned_periods = planned_yielding_periods
    max_reward_per_period = total_rewards / planned_periods
    now_period = get_now_period(params, blocks_per_period)
    pool_id = get_next_id(params)

    pool = GlobalPool(pool_id, now_period, reward_currency, yield_per_period, planned_yielding_periods,
                      blocks_per_period, owner, incentivized_token, max_reward_per_period)

    GlobalPoolData = params['GlobalPoolData']
    LiquidityPoolData = params['LiquidityPoolData']

    GlobalPoolData.append(pool)
    LiquidityPoolData.append({})

    pool.free_balance += total_rewards


def destroy_farm(params, origin, farm_id):  # QUESTION: does this need to push out unclaimed but earned rewards?
    GlobalPoolData = params['GlobalPoolData']
    LiquidityPoolData = params['LiquidityPoolData']
    GlobalPoolData[farm_id] = None
    LiquidityPoolData[farm_id] = {}


def withdraw_undistributed_rewards(origin, farm_id):
    pass


def add_liquidity_pool(params, origin, farm_id, asset_pair, weight, loyalty_curve=None):
    who = origin
    assert weight != 0
    GlobalPoolData = params['GlobalPoolData']
    g_pool = GlobalPoolData[farm_id]
    assert who == g_pool.owner
    amm_pool_id = asset_pair  # simplification for now, for modeling purposes
    LiquidityPoolData = params['LiquidityPoolData']
    assert amm_pool_id not in LiquidityPoolData[farm_id]

    now_period = get_now_period(params, g_pool.blocks_per_period)

    update_global_pool(g_pool, now_period)
    g_pool.liq_pools_count += 1

    liq_pool_id = farm_id + "_" + amm_pool_id
    pool = LiquidityPool(liq_pool_id, now_period, loyalty_curve, weight)
    LiquidityPoolData[farm_id][amm_pool_id] = pool


def update_liquidity_pool(old_state, block_number, origin, farm_id, asset_pair, weight):
    who = origin
    assert weight != 0

    amm_pool_id = asset_pair
    LiquidityPoolData = old_state['LiquidityPoolData']
    liq_pool = LiquidityPoolData[farm_id][amm_pool_id]
    GlobalPoolData = old_state['GlobalPoolData']
    g_pool = GlobalPoolData[farm_id]

    assert g_pool.owner == who

    now_period = get_period_number(block_number, g_pool.blocks_per_period)

    update_global_pool(g_pool, now_period)

    pool_reward = claim_from_global_pool(g_pool, liq_pool.stake_in_global_pool)
    update_pool(old_state, liq_pool, pool_reward, now_period, g_pool.id, g_pool.reward_currency)

    incentivized_token_balance_in_amm = liq_pool.incentivized_token_in_amm
    new_stake_in_global_pool = incentivized_token_balance_in_amm * liq_pool.total_shares * weight

    if new_stake_in_global_pool > liq_pool.stake_in_global_pool:
        diff = new_stake_in_global_pool - liq_pool.stake_in_global_pool
        g_pool.total_shares += diff
    else:
        diff = liq_pool.stake_in_global_pool - new_stake_in_global_pool
        g_pool.total_shares -= diff

    liq_pool.stake_in_global_pool = new_stake_in_global_pool
    liq_pool.multiplier = weight


def cancel_liquidity_pool(origin, farm_id):
    pass


def remove_liquidity_pool(origin, farm_id):
    pass


# why is this "deposit shares"? Are users going to need to deposit their LP shares somewhere to get LM rewards?
# aren't they just going to get rewards automatically for having liquidity contributed to the pool?
# then this should be "deposit_liquidity" or something
def deposit_shares(old_state, block_number, origin, farm_id, asset_pair, amount):
    who = origin
    #amm_share = get_share_token(asset_pair)

    liq_pool_key = asset_pair
    LiquidityPoolData = old_state['LiquidityPoolData']
    liq_pool = LiquidityPoolData[farm_id][liq_pool_key]
    update_liquidity_pool(params, origin, farm_id, asset_pair, liq_pool.multiplier + amount)  # are "weight" and "multiplier" the same?

    # return position NFT... although it is actually semi-fungible
    return {'owner': origin, 'farm_id': farm_id, 'pool': asset_pair, 'amount': amount, 'block_deposited': block_number,
            'accumulated_rps_start': liq_pool.accumulated_rps, 'accumulated_claimed_rewards': 0}


def claim_rewards(params, position):
    pool = params['LiquidityPoolData'][position['pool']]

    user_accumulated_rps = position['accumulated_rps_start']
    user_shares = position['amount']
    accumulated_rps_now = pool.accumulated_rps
    user_accumulated_claimed_rewards = position['accumulated_claimed_rewards']
    farm_id = position['farm_id']
    blocks_per_period = params['GlobalFarmData'][farm_id].blocks_per_period
    periods = get_period_number(params['T'], blocks_per_period) - get_period_number(position['block_deposited'], blocks_per_period)
    loyalty_curve = pool.loyalty_curve
    loyalty_multiplier = get_loyalty_multiplier(periods, loyalty_curve)
    rewards, locked_rewards = get_user_reward(user_accumulated_rps, user_shares, accumulated_rps_now,
                                              user_accumulated_claimed_rewards, loyalty_multiplier)
    position['accumulated_claimed_rewards'] += rewards
    pool.free_balance -= rewards
    return locked_rewards


def withdraw_shares(params, position):
    lost_rewards = claim_rewards(params, position)

    liq_pool_key = position['pool']
    farm_id = position['farm_id']
    LiquidityPoolData = params['LiquidityPoolData']
    liq_pool = LiquidityPoolData[farm_id][liq_pool_key]
    g_pool = params['GlobalPoolData'][farm_id]

    update_liquidity_pool(params, position['owner'], farm_id, liq_pool_key, liq_pool.multiplier - position['amount'])
    position['amount'] = 0
    liq_pool.free_balance -= lost_rewards
    g_pool.free_balance += lost_rewards


def get_next_id(params):
    return len(params['GlobalPoolData']) + 1


def get_period_number(block, blocks_per_period):
    return int(block / blocks_per_period)


def set_now_block(block_number):
    """Called from cadCAD to update block number"""
    T = block_number


def get_loyalty_multiplier(periods: int, curve: LoyaltyCurve):
    if curve.initial_reward_percentage == 1:
        return 1

    denom = (curve.initial_reward_percentage + 1) * curve.scale_coef
    p = periods
    theta = p/denom
    theta_mul_b = theta * curve.initial_reward_percentage
    theta_add_theta_mul_b = theta + theta_mul_b
    num = theta_add_theta_mul_b + curve.initial_reward_percentage
    denom = theta_add_theta_mul_b + 1

    return num/denom


def get_reward_per_period(yield_per_period: float, total_global_farm_shares: float, max_reward_per_period: float) -> float:
    return min(yield_per_period * total_global_farm_shares, max_reward_per_period)


def update_global_pool(pool: GlobalPool, now_period: int):
    if pool.updated_at == now_period:
        pass
    if pool.total_shares == 0:
        pass
    periods_since_last_update = now_period - pool.updated_at
    reward_per_period = get_reward_per_period(pool.yield_per_period, pool.total_shares, pool.max_reward_per_period)
    reward = min(periods_since_last_update * reward_per_period, pool.free_balance)
    if reward != 0:
        pool.accumulated_rps = get_accumulated_rps(pool.accumulated_rps, pool.total_shares, reward)
        pool.accumulated_rewards += reward

    pool.updated_at = now_period


def get_accumulated_rps(accumulated_rps_now: float, total_shares: float, reward: float) -> float:
    return reward/total_shares + accumulated_rps_now


# can we name 'user_accumulated_rps' more carefully? should be rps of liquidity pool at point of deposit?
def get_user_reward(user_accumulated_rps: float, user_shares: float, accumulated_rps_now: float,
                    user_accumulated_claimed_rewards: float, loyalty_multiplier: float) -> (float, float):
    max_rewards = (accumulated_rps_now - user_accumulated_rps) * user_shares
    claimable_rewards = loyalty_multiplier * max_rewards
    unclaimable_rewards = max_rewards - claimable_rewards
    user_rewards = claimable_rewards - user_accumulated_claimed_rewards
    return (user_rewards, unclaimable_rewards)


def claim_from_global_pool(pool: GlobalPool, shares: float):
    reward = min((pool.accumulated_rps - pool.accumulated_rps_start) * shares, pool.accumulated_rewards)
    pool.accumulated_rps_start = pool.accumulated_rps
    pool.paid_accumulated_rewards += reward
    pool.accumulated_rewards -= reward
    return reward


def update_pool(params, pool, rewards, period_now, global_pool_id, reward_currency):
    if pool.updated_at == period_now:
        return
    if pool.total_shares == 0:
        return

    pool.accumulated_rps = get_accumulated_rps(pool.accumulated_rps, pool.total_shares, rewards)
    pool.updated_at = period_now

    GlobalPoolData = params['GlobalPoolData']
    global_pool_balance = GlobalPoolData[global_pool_id].free_balance

    assert global_pool_balance >= rewards

    GlobalPoolData[global_pool_id].free_balance -= rewards
    pool.free_balance += rewards
