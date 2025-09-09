from matplotlib import pyplot as plt
import sys, os
import streamlit as st
import pandas as pd


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, balance_ratio_at_price
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule

susde_price = 1.19
susds_price = 1.06
pegs = {'aUSDT': 1, 'aUSDC': 1, 'sUSDS': susds_price, 'sUSDE': susde_price}
hsm_liquidity = {'aUSDT': 100000, 'aUSDC': 100000, 'sUSDS': int(50000/susds_price), 'sUSDE': int(50000/susde_price)}
# initial_tvl = 1000000
sell_price_fee = 0.0
default_amp = 1000
default_hours_to_simulate = 24.0
default_hours_to_simulate /= 100

stableswap_liq = {tkn: {'HOLLAR': 250000, tkn: 250000/pegs[tkn]} for tkn in hsm_liquidity}

st.sidebar.header("Parameters")
amp_dict = {tkn: [1000] for tkn in hsm_liquidity}
df = pd.DataFrame(amp_dict, index=["amp"])
edited_df = st.sidebar.data_editor(df, use_container_width=True)
amps = edited_df.loc["amp"].to_dict()

buyback_speed_init = {tkn: [0.0001] for tkn in hsm_liquidity}
df = pd.DataFrame(buyback_speed_init, index=["buyback speed"])
edited_df = st.sidebar.data_editor(df, use_container_width=True)
buyback_speeds = edited_df.loc["buyback speed"].to_dict()
st.sidebar.text("Buyback speed is the percentage of the pool imbalance that can be corrected by HSM in one block.")

buy_coef_init = {tkn: [0.995] for tkn in hsm_liquidity}
df = pd.DataFrame(buy_coef_init, index=["max buy price"])
edited_df = st.sidebar.data_editor(df, use_container_width=True)
max_buy_prices = edited_df.loc["max buy price"].to_dict()
st.sidebar.text("The max buy price coefficient times the peg is the highest price at which the HSM will buy Hollar.")

init_price = st.sidebar.number_input(
    "Initial price of Hollar in USDT",
    min_value=0.001, max_value=1.0, value=0.5, step=0.001, key="init_price", format="%.3f"
)

st.sidebar.markdown("---")
hours = st.sidebar.number_input(
    "Hours to simulate",
    min_value=0.000001, max_value=1000.0, value=default_hours_to_simulate, step=1.0, key="hours"
)
blocks_per_hour = st.sidebar.number_input(
    "Blocks per hour",
    min_value=300, max_value=1800, value=600, step=1, key="blocks_per_hour"
)
blocks = hours * blocks_per_hour
st.sidebar.markdown("---")



buy_fee = 0.0001
pools = {}
for tkn in hsm_liquidity:
    hol_pct = balance_ratio_at_price(amps[tkn], init_price)
    initial_tvl = stableswap_liq[tkn]['HOLLAR'] * 2
    init_hollar = initial_tvl * hol_pct
    tokens = {'HOLLAR': init_hollar, tkn: initial_tvl * (1 - hol_pct)}
    pools[tkn] = StableSwapPoolState(tokens=tokens, amplification=amps[tkn], trade_fee=0.0004)
hsm = StabilityModule(
    liquidity = hsm_liquidity,
    buyback_speed = [buyback_speeds[tkn] for tkn in hsm_liquidity],
    pools = [pools[tkn] for tkn in hsm_liquidity],
    sell_price_fee = sell_price_fee,
    max_buy_price_coef = [max_buy_prices[tkn] for tkn in hsm_liquidity],
    buy_fee = buy_fee
)
agent = Agent(holdings={'HOLLAR': 0, 'USDT': 0})

buy_amts = {tkn: [] for tkn in hsm_liquidity}
buy_prices = {tkn: [] for tkn in hsm_liquidity}
buy_spots = {tkn: [] for tkn in hsm_liquidity}
hsm_liquidity_history = {tkn: [] for tkn in hsm_liquidity}
profits = {tkn: [] for tkn in hsm_liquidity}

for i in range(int(blocks)):
    for tkn in hsm_liquidity:
        pool = pools[tkn]
        buy_spots[tkn].append(pool.buy_spot(tkn_buy='HOLLAR', tkn_sell=tkn))
        buy_amt, buy_price = hsm.get_buy_params(tkn=tkn)
        sell_amt = pool.calculate_sell_from_buy(tkn_buy='HOLLAR', tkn_sell=tkn, buy_quantity=buy_amt)
        hsm.arb(agent, tkn=tkn)
        buy_amts[tkn].append(buy_amt)
        buy_prices[tkn].append(buy_price)
        hsm_liquidity_history[tkn].append(hsm.liquidity[tkn])
        hsm.update()
        profits[tkn].append(agent.holdings[tkn])

# hollar_burned = init_hollar - pool.liquidity['HOLLAR']

# Plot 1: max buy amounts from stability module
fig1, ax1 = plt.subplots()
for tkn in hsm_liquidity:
    ax1.plot(buy_amts[tkn], label=tkn)
ax1.set_title("max buy amounts from stability module")
ax1.set_xlabel("blocks")
ax1.set_ylabel("Hollar")
ax1.legend()
st.pyplot(fig1)



# Plot 4: stability module liquidity
fig4, ax4 = plt.subplots()
for tkn in hsm_liquidity_history:
    ax4.plot(hsm_liquidity_history[tkn], label=tkn)
ax4.set_xlabel("blocks")
ax4.set_title("stability module liquidity")
ax4.legend()
st.pyplot(fig4)

# Plot 5: buy spot price vs USDT spent
for tkn in hsm_liquidity_history:
    tkn_spent = [1000000 - hsm_liquidity for hsm_liquidity in hsm_liquidity_history[tkn]]
    fig5, ax5 = plt.subplots()
    ax5.plot(tkn_spent, buy_spots[tkn])
    ax5.set_title("buy spot price vs " + tkn + " spent")
    ax5.set_xlabel(tkn + " spent")
    st.pyplot(fig5)

# Plot 6: profits
fig6, ax6 = plt.subplots()
ax6.plot(profits['sUSDS'])
ax6.set_title("arbitrager profits")
ax6.set_xlabel("blocks")
ax6.set_ylabel("USDT")
st.pyplot(fig6)

# Plot 3: price difference between HSM and pool
fig3, ax3 = plt.subplots()
price_ratios = [(buy_price / buy_spot) - 1 for buy_price, buy_spot in zip(buy_prices['sUSDS'], buy_spots['sUSDS']) if buy_price > 0]
ax3.plot(price_ratios)
ax3.set_xlabel("blocks")
ax3.set_title("HSM buy price / pool spot price - 1")
st.pyplot(fig3)

# Plot 2: stability module buy prices
fig2, ax2 = plt.subplots()
for tkn in buy_prices:
    ax2.plot(buy_prices['sUSDS'], label=tkn + ' HSM prices')
    ax2.plot(buy_spots['sUSDS'], label=tkn + ' pool prices', linestyle='--')
ax2.set_title("Prices")
ax2.set_xlabel("blocks")
ax2.legend()
st.pyplot(fig2)