import copy

from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.omnipool_amm import OmnipoolState, trade_to_price
from hydradx.model.amm.agents import Agent

def run_app():
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
        preferred_stablecoin='USD'
    )

    max_buy = initial_omnipool.liquidity['HDX'] * 1.5
    max_sell = initial_omnipool.liquidity['HDX'] * 0.5
    steps = 30
    buys = [max_buy / 1.5 ** i for i in range(steps // 2)]
    sells = [max_sell / 1.5 ** i for i in range(steps // 2)]
    trades = [-buy for buy in buys] + sells[::-1]
    trade_agent = Agent(enforce_holdings=False)
    initial_hdx_price = initial_omnipool.lrna_price('HDX')

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

    for trade in trades:
        omnipool = initial_omnipool.copy()
        pool_agent = Agent(enforce_holdings=False)
        trade_agent.holdings = {}
        omnipool.add_liquidity(
            agent=pool_agent,
            tkn_add='HDX',
            quantity=omnipool.liquidity['HDX']
        )
        initial_shares = pool_agent.get_holdings(('omnipool', 'HDX'))
        initial_hdx = pool_agent.get_holdings('HDX')
        pool_agent.holdings['HDX'] = 0
        initial_value = omnipool.cash_out(pool_agent)

        # trade in one direction
        if trade < 0:
            omnipool.swap(agent=trade_agent, tkn_sell='HDX', tkn_buy='LRNA', sell_quantity=-trade)
        else:
            omnipool.swap(agent=trade_agent, tkn_buy='HDX', tkn_sell='LRNA', buy_quantity=trade)
        price_change = (omnipool.lrna_price('HDX') - initial_hdx_price) / initial_hdx_price
        # remove and re-add liquidity
        remove_readd(omnipool, pool_agent)
        # swap back
        trade_back = trade_to_price(omnipool, 'HDX', initial_hdx_price)
        if trade_back > 0:
            omnipool.swap(
                agent=trade_agent, tkn_sell='HDX', tkn_buy='LRNA', sell_quantity=trade_back
            )
        elif trade_back < 0:
            omnipool.swap(
                agent=trade_agent, tkn_sell='LRNA', tkn_buy='HDX', buy_quantity=-trade_back
            )
        remove_readd(omnipool, pool_agent)
        trade_back = trade_to_price(omnipool, 'HDX', initial_hdx_price)
        if trade_back > 0:
            omnipool.swap(
                agent=trade_agent, tkn_sell='HDX', tkn_buy='LRNA', sell_quantity=trade_back
            )
        elif trade_back < 0:
            omnipool.swap(
                agent=trade_agent, tkn_sell='LRNA', tkn_buy='HDX', buy_quantity=-trade_back
            )
        if abs(omnipool.lrna_price('HDX') - initial_hdx_price) / initial_hdx_price > 0.00000001:
            pass
        loss = (pool_agent.get_holdings(('omnipool', 'HDX')) - initial_shares) / initial_shares
        final_value = omnipool.cash_out(pool_agent)
        # initial_value = omnipool.lrna_price('HDX')
        loss = (final_value - initial_value) / initial_value
        print(f"price change {round(price_change * 100, 3)}%, loss {round(loss * 100, 3)}%")
run_app()
