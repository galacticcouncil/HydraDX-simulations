from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent

DEFAULT_AMP = 100

amp = st.number_input(
    "Amplification for stableswap pools",
    min_value=5, value=DEFAULT_AMP, step=1, key="amp"
)
max_trade_pct = st.number_input(
    "Max Hollar trade size, as percentage of Hollar in initial pool",
    min_value=0.0, value=0.9, step=0.1, key="max_trade_size"
)

liquidity = {'HOLLAR': 250000, 'aUSDT': 250000}
peg = 1

trade_pcts = [i * max_trade_pct / 100 for i in range(1, 101)]
execution_prices = []
spot_prices = []
ratio = []
for trade_pct in trade_pcts:
    pool = StableSwapPoolState(tokens=liquidity,
                               amplification=amp,
                               trade_fee=0.0004,
                               peg=peg)

    alice = Agent(enforce_holdings=False)
    trade_size = trade_pct * liquidity['HOLLAR']
    pool.swap(alice, tkn_buy='HOLLAR', tkn_sell='aUSDT', buy_quantity=trade_size)
    exec_price = -alice.get_holdings('aUSDT') / alice.get_holdings('HOLLAR')
    spot_buy = pool.buy_spot('HOLLAR', 'aUSDT', 0.0)
    execution_prices.append(exec_price)
    spot_prices.append(spot_buy)
    ratio.append(pool.liquidity['HOLLAR'] / (pool.liquidity['HOLLAR'] + pool.liquidity['aUSDT']))

# graph execution prices and spot prices

fig, ax = plt.subplots()
ax.plot(ratio, spot_prices)
ax.set_title("spot price after trade")
ax.set_xlabel("Hollar % of pool assets after trade")
st.pyplot(fig)

fig, ax = plt.subplots()
ax.plot(ratio, execution_prices)
ax.set_title("execution price of trade")
ax.set_xlabel("Hollar % of pool assets after trade")
st.pyplot(fig)
