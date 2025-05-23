import copy
import math

from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, simulate_swap
from hydradx.model.amm.agents import Agent
from hydradx.apps.gigadot_modeling.utils import get_omnipool_minus_vDOT, set_up_gigaDOT_3pool, set_up_gigaDOT_2pool, \
    create_custom_scenario, simulate_route, get_slippage_dict
from hydradx.apps.gigadot_modeling.display_utils import display_liquidity, display_op_and_ss
from hydradx.model.indexer_utils import get_latest_stableswap_data

trade_fee_input = st.number_input(
    "Swap fee, in percent",
    min_value=0.0, value=0.02, key="trade_fee", format="%.4f",
    help="An integer here is a percentage, so 2 means 2%, while 0.02 is 0.02%, or 2 bps."
)
trade_fee = trade_fee_input / 100

col1, col2 = st.columns(2)
with col1:
    amp_start = st.number_input(
        "GigaDOT amplification start",
        min_value=10, value=22, step=1, key="amp_start"
    )
    gigadot_adot = st.number_input(
        "GigaDOT aDOT amount",
        min_value=0, value=1097138, step=1, key="gigadot_adot"
    )
with col2:
    amp_end = st.number_input(
        "GigaDOT amplification end",
        min_value=10, value=100, step=1, key="amp_end"
    )
    gigadot_vdot = st.number_input(
        "GigaDOT vDOT amount",
        min_value=0, value=757890, step=1, key="gigadot_vdot"
    )
max_trade_size = st.number_input(
    "Max trade size, in bought (vDOT)",
    min_value=2, value=10000, step=1, key="max_trade_size"
)

gigadot_liquidity = {"aDOT": gigadot_adot, "vDOT": gigadot_vdot}
peg = 1097138.414 / 757890.773

slippage_dict = {}

init_pool = StableSwapPoolState(
    tokens=gigadot_liquidity,
    amplification=amp_start,
    trade_fee=trade_fee,
    peg=peg
)
pool = init_pool.copy()

for amp in [amp_start, amp_end]:

    pool.amplification = amp
    n = 10
    min_trade_size = 1
    step = (max_trade_size - min_trade_size) / (n - 1) if n > 1 else 0
    buy_sizes = [min_trade_size + i * step for i in range(n)]

    agent = Agent(enforce_holdings=False)
    sell_amts = []

    for buy_size in buy_sizes:
        new_pool, new_agent = simulate_swap(pool, agent, tkn_sell='aDOT', tkn_buy='vDOT', buy_quantity=buy_size)
        sell_amts.append(-new_agent.get_holdings('aDOT'))


    # graph slippage
    prices = [sell_amts[i] / buy_sizes[i] for i in range(len(sell_amts))]
    lowest_price = prices[0]
    slippage_dict[amp] = [(prices[i] - lowest_price) / lowest_price for i in range(len(sell_amts))]

fig, ax = plt.subplots(figsize=(10, 6))
for amp in slippage_dict:
    ax.plot(buy_sizes, slippage_dict[amp], label=f"amp {amp}")
ax.legend()
ax.set_title(f"Slippage")
ax.set_xlabel('Buy size')
ax.set_ylabel('Slippage')
st.pyplot(fig)

pool = init_pool.copy()



amps = list(range(amp_start, amp_end + 1))
prices = []
for amp in amps:
    pool.amplification = amp
    prices.append(pool.price('vDOT', 'aDOT'))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(amps, prices, label="vDOT price")
ax.legend()
ax.set_title(f"vDOT price")
ax.set_xlabel('Amplification')
ax.set_ylabel('vDOT price in aDOT')
st.pyplot(fig)


# model exploit
# buy asset pool has too much of, sell it back to the pool after the A adjustment
# first identify which asset is oversupplied
vdot_value = gigadot_vdot * peg
if vdot_value > gigadot_adot:
    # vDOT is oversupplied
    tkn_buy = 'aDOT'
    tkn_sell = 'vDOT'
else:
    # aDOT is oversupplied
    tkn_buy = 'vDOT'
    tkn_sell = 'aDOT'

col1, col2 = st.columns(2)
with col1:
    amp_change_per_block = st.number_input(
        "Amplification change per block",
        min_value=1, value=1, key="amp_change_per_block"
    )
with col2:
    attack_size = st.number_input(
        "Attack buy size",
        min_value=0.01, value=100.0, key="attack_size"
    )

amps = list(range(amp_start, amp_end + 1, amp_change_per_block))
if amps[-1] != amp_end:
    amps.append(amp_end)
agent = Agent(enforce_holdings=False)
profit = []
for amp in amps[1:]:
    pool.swap(agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=attack_size)
    if pool.fail:
        raise ValueError
    pool.amplification = amp
    pool.swap(agent, tkn_buy=tkn_sell, tkn_sell=tkn_buy, sell_quantity=attack_size)
    if pool.fail:
        raise ValueError
    profit.append(agent.get_holdings(tkn_sell))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(amps[1:], profit, label="attacker profit")
ax.legend()
ax.set_title(f"Attacker profit in {tkn_sell}")
ax.set_xlabel('Amplification')
st.pyplot(fig)

#
# # Define markers for better visibility
# markers = ['o', 's', '^', 'D', 'x']  # Circle, square, triangle, diamond, cross
#
# # Mapping token pairs to marker indices
# for idx, tkn_pair in enumerate([('DOT', 'vDOT'), ('2-Pool', 'vDOT'), ('2-Pool', 'DOT')]):
#     st.subheader(f"Options for {tkn_pair} Slippage")
#
#     options = ["Current", "GigaDOT as 3-pool", "GigaDOT as 2-pool", "Custom Scenario"]
#     # The multiselect returns a list of selected series; by default, all are selected.
#     selected_series = st.multiselect(f"Select series to display for {tkn_pair}", options, default=options, key=f"multiselect_{tkn_pair}")
#
#     # Create the figure and axis for this graph
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # Conditionally plot each series based on checkbox values
#     if "Current" in selected_series:
#         n = len(slippage["Current Omnipool"][tkn_pair])
#         ax.plot(buy_sizes[0:n], slippage["Current Omnipool"][tkn_pair],
#                 marker=markers[0], linestyle='-', label='Current', linewidth=2)
#     if "GigaDOT as 3-pool" in selected_series:
#         n = len(slippage["gigaDOT with 3 assets"][tkn_pair])
#         ax.plot(buy_sizes[0:n], slippage["gigaDOT with 3 assets"][tkn_pair],
#                 marker=markers[1], linestyle='--', label='GigaDOT as 3-pool', linewidth=2)
#     if "GigaDOT as 2-pool" in selected_series:
#         n = len(slippage["gigaDOT with 2 assets"][tkn_pair])
#         ax.plot(buy_sizes[0:n], slippage["gigaDOT with 2 assets"][tkn_pair],
#                 marker=markers[4], linestyle='-', label='GigaDOT as 2-pool', linewidth=2)
#     if "Custom Scenario" in selected_series and "Custom Scenario" in slippage:
#         n = len(slippage["Custom Scenario"][tkn_pair])
#         ax.plot(buy_sizes[0:n], slippage["Custom Scenario"][tkn_pair],
#                 marker=markers[2], linestyle='--', label='Custom', linewidth=2)
#
#     # Add grid, legend, title, and axis labels
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.legend()
#     ax.set_title(f"{tkn_pair} Slippage")
#     ax.set_xlabel('Buy size')
#     ax.set_ylabel('Slippage')
#
#     # Display the plot in Streamlit
#     st.pyplot(fig)
