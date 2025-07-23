from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent


st.sidebar.text(
    "Stableswap amplification is 100."
)

amp = 100
liquidity = {'aETH': 1451, 'wstETH': 1355}
peg = 1.20780546
pool = StableSwapPoolState(tokens=liquidity,
                           amplification=amp,
                           trade_fee=0.00069,
                           unique_id='GETH',
                           peg=peg)
wsteth_spot_feeless = pool.buy_spot('wstETH', 'aETH', 0)
wsteth_spot_buy = pool.buy_spot('wstETH', 'aETH')
wsteth_spot_sell = pool.sell_spot('aETH', 'wstETH')
geth_spot = pool.buy_shares_spot('aETH')
geth_spot2 = pool.buy_shares_spot('wstETH')
geth_spot3 = pool.remove_liquidity_spot('aETH')

alice = Agent(enforce_holdings=False)
pool.swap(alice, tkn_buy='wstETH', tkn_sell='aETH', buy_quantity=130)

wsteth_spot_feeless = pool.buy_spot('wstETH', 'aETH', 0)
wsteth_spot_buy = pool.buy_spot('wstETH', 'aETH')
wsteth_spot_sell = pool.sell_spot('aETH', 'wstETH')
geth_spot = pool.buy_shares_spot('aETH')
geth_spot2 = pool.buy_shares_spot('wstETH')
geth_spot3 = pool.remove_liquidity_spot('aETH')
print(pool)