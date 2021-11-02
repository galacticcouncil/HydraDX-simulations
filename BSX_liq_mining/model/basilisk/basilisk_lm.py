import copy
import math
import string


class LoyaltyCurve:
    def __init__(self, b=0.5, scale_coef=100):
        self.b = b
        self.scale_coef = scale_coef


class GlobalPool:
    def __init__(self, starting_block: int, reward_currency: string):
        self.updated_at = starting_block
        self.reward_currency = reward_currency
        self.total_shares = 0
        self.accumulated_rps_start = 0
        self.accumulated_rps = 0
        self.accumulated_rewards = 0
        self.paid_accumulated_rewards = 0
        self.free_balance = 0


def create_new_program(origin, pool_id, currency_id, loyalty_curve):
    pass  # Todo


def get_period_number(now: int, accumulate_period: int):
    return now/accumulate_period  # Todo: should this be floored or something?


def get_loyalty_multiplier(periods: int, curve: LoyaltyCurve):
    if curve.b == 1:
        return 1

    denom = (curve.b + 1) * curve.scale_coef
    p = periods
    theta = p/denom
    theta_mul_b = theta * curve.b
    theta_add_theta_mul_b = theta + theta_mul_b
    num = theta_add_theta_mul_b + curve.b
    denom = theta_add_theta_mul_b + 1

    return num/denom


def get_reward_per_period(yield_per_period: float, total_global_farm_shares: float, max_reward_per_period: float) -> float:
    return min(yield_per_period * total_global_farm_shares, max_reward_per_period)


def update_global_pool(pool_id, pool: GlobalPool, now_period: int, reward_per_period: float):
    if pool.updated_at == now_period:
        pass
    if pool.total_shares == 0:
        pass
    periods_since_last_update = now_period - pool.updated_at
    #reward = min(periods_since_last_update * reward_per_period, free_balance(pool.reward_currency, pool_id))  # Todo: do we need this in the modeling?
    reward = periods_since_last_update * reward_per_period
    if reward != 0:
        pool.accumulated_rps = get_new_accumulated_rps(pool.accumulated_rps, pool.total_shares, reward)
        pool.accumulated_rewards += reward


def get_new_accumulated_rps(accumulated_rps_now: float, total_shares: float, reward: float) -> float:
    return reward/total_shares + accumulated_rps_now


def get_user_reward(user_accumulated_rps: float, user_shares: float, accumulated_rps_now: float, user_accumulated_claimed_rewards: float, loyalty_multiplier: float) -> (float, float):
    # Todo fix the return type to pair of floats
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
