import copy

from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState, trade_to_price as get_trade_to_price
from hydradx.model.amm.agents import Agent

def run_app():
    print()
    low_price = 0.98
    high_price = 1.02
    steps = 30
    buys = [1 - (1 - low_price) / 1.5 ** i for i in range(steps // 2)]
    sells = [1 + (high_price - 1) / 1.5 ** i for i in range(steps // 2)]
    prices = buys + [1] + sells[::-1]
    lp_liquidity_pct = 0.25
    print(f"LP liquidity pct: {lp_liquidity_pct * 100}%")
    initial_omnipool = OmnipoolState(
        tokens={
            'HDX': {'liquidity': 44000000, 'LRNA': 44000000},
            'WETH': {'liquidity': 1400, 'LRNA': 2276599},
            'USD': {'liquidity': 1, 'LRNA': 1},
            'DAI': {'liquidity': 22682620000000, 'LRNA': 22682620000000},
            'DOT': {'liquidity': 88000, 'LRNA': 546461},
            'WBTC': {'liquidity': 47, 'LRNA': 1145210},
        },
        withdrawal_fee=False,
        lrna_fee_burn=0.5,
        lrna_mint_pct=1.0,
        asset_fee=0.5,
        lrna_fee=0.5,
        preferred_stablecoin='USD'
    )
    print(f"asset fee: {initial_omnipool.asset_fee('HDX') * 100}%, lrna fee: {initial_omnipool.lrna_fee('HDX') * 100}%")

    def trade_to_price(pool: OmnipoolState, agent: Agent, tkn: str, price: float):
        trade_size = get_trade_to_price(pool, 'HDX', price)
        if trade_size > 0:
            pool.swap(
                agent=agent, tkn_sell=tkn, tkn_buy='LRNA', sell_quantity=trade_size
            )
        elif trade_size < 0:
            # this one gets tricky
            trade_size *= 1 - pool.asset_fee(tkn)
            pool.swap(
                agent=agent, tkn_sell='LRNA', tkn_buy=tkn, buy_quantity=-trade_size
            )
            pool_price = pool.lrna_price(tkn)
            if abs(pool_price - price) / price > 0.00000001:
                raise ValueError(
                    f"Trade to price failed: {pool_price} != {price} for {tkn}"
                )

    def remove_readd(pool: OmnipoolState, agent: Agent):
        pool.remove_liquidity(
            agent=agent,
            tkn_remove='HDX'
        )
        if agent.get_holdings('LRNA') > 0:
            pool.swap(agent=agent, tkn_sell='LRNA', tkn_buy='HDX', sell_quantity = agent.get_holdings('LRNA'))
        pool.add_liquidity(
            agent=agent,
            tkn_add='HDX',
            quantity=agent.get_holdings('HDX')
        )

    trade_agent = Agent(enforce_holdings=False)
    initial_hdx_price = initial_omnipool.lrna_price('HDX')

    for price in prices:
        omnipool = initial_omnipool.copy()
        pool_agent = Agent(enforce_holdings=False)
        trade_agent.holdings = {}
        omnipool.add_liquidity(
            agent=pool_agent,
            tkn_add='HDX',
            quantity=omnipool.liquidity['HDX'] * lp_liquidity_pct / (1 - lp_liquidity_pct)
        )
        # print(pool_agent.holdings[('omnipool', 'HDX')] / omnipool.shares['HDX'])
        initial_shares = pool_agent.get_holdings(('omnipool', 'HDX'))
        initial_hdx_liquidity = omnipool.liquidity['HDX']
        pool_agent.holdings['HDX'] = 0
        initial_value = omnipool.cash_out(pool_agent)

        trade_to_price(omnipool, trade_agent, 'HDX', price)
        price_change = (omnipool.lrna_price('HDX') - initial_hdx_price) / initial_hdx_price
        # remove and re-add liquidity
        remove_readd(omnipool, pool_agent)
        # swap back
        trade_to_price(omnipool, trade_agent,'HDX', initial_hdx_price)
        # print("price before remove/readd:", round(omnipool.lrna_price('HDX'), 6))
        remove_readd(omnipool, pool_agent)
        trade_to_price(omnipool, trade_agent, 'HDX', initial_hdx_price)
        final_value = omnipool.cash_out(pool_agent)
        # initial_value = omnipool.lrna_price('HDX')
        loss = (final_value - initial_value) / initial_value
        print(f"{loss * 100}%")
        end_price = omnipool.lrna_price('HDX')
        end_liquidity = omnipool.lrna['HDX']
        print(end_liquidity)
        # print(f"end price: {round(end_price, 6)}")
        # print(f"price change {round(price_change * 100, 3)}%, loss {round(loss * 100, 10)}%")
run_app()
