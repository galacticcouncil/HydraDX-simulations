from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.omnipool_amm import OmnipoolState, simulate_swap, simulate_remove_liquidity, simulate_add_liquidity
from hydradx.model.amm.agents import Agent

liquidity = {
    'HDX': {'liquidity': 1000000, 'LRNA': 1000000},
    'DOT': {'liquidity': 1000000, 'LRNA': 1000000},
    'USD': {'liquidity': 1000000, 'LRNA': 1000000},
}

omnipool = OmnipoolState(tokens=liquidity, preferred_stablecoin='USD')

trade_size = 100000
agent1 = Agent(holdings={'DOT': 100000})
agent2 = Agent(holdings={'DOT': 100000})
omnipool.add_liquidity(agent1, agent1.get_holdings('DOT'), 'DOT')
trade_agent = Agent(enforce_holdings=False)
omnipool.swap(trade_agent, tkn_buy='HDX', tkn_sell='DOT', buy_quantity=trade_size)  # move price
omnipool.add_liquidity(agent2, agent2.get_holdings('DOT'), 'DOT')

# two LPs now have LP positions of same initial DOT at different prices
# now we look at their relative value at different price points



trade_sizes = [i * 10000 for i in range(11)]
dot_prices = []
agent1_value = []
agent2_value = []
for trade_size in trade_sizes:
    temp_state, temp_agent = simulate_swap(omnipool, trade_agent, tkn_buy='HDX', tkn_sell='DOT', sell_quantity=trade_size)
    prices_after_trade = {tkn: temp_state.price(tkn, 'LRNA') for tkn in liquidity}
    dot_prices.append(prices_after_trade['DOT'])
    agent1_value.append(temp_state.cash_out(agent1, prices_after_trade))
    agent2_value.append(temp_state.cash_out(agent2, prices_after_trade))

# initial_pos_values = [100000 * x for x in dot_prices]
# pos_value = [cash_value[i]/initial_pos_values[i] for i in range(len(initial_pos_values))]
# loop_val_ratio = [loop_value[i]/initial_pos_values[0] for i in range(len(initial_pos_values))]


fig1, ax1 = plt.subplots()
ax1.plot(dot_prices, agent1_value, label='Agent 1')
ax1.plot(dot_prices, agent2_value, label='Agent 2')
ax1.legend()
ax1.set_ylabel("position value")
ax1.set_xlabel("dot price")
st.pyplot(fig1)

agent_val_ratio = [agent1_value[i] / agent2_value[i] for i in range(len(agent1_value))]
agent_val_ratio_reversed = [agent2_value[i] / agent1_value[i] for i in range(len(agent2_value))]
fig2, ax2 = plt.subplots()
ax2.plot(dot_prices, agent_val_ratio, label='Agent 1 / Agent 2')
ax2.plot(dot_prices, agent_val_ratio_reversed, label='Agent 2 / Agent 1')
ax2.legend()
ax2.set_xlabel("dot price")
ax2.set_ylabel("agent value ratio")
st.pyplot(fig2)