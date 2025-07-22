from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, balance_ratio_at_price
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule

st.sidebar.text(
    "Initial HSM liquidity is 1 million USDT. Initial stableswap pool is USDT/Hollar pool with 1 million total tokens"
    " (USDT and Hollar combined). A is 100."
)

hsm_liquidity = {'USDT': 1000000}
# initial_tvl = 1000000
sell_price_fee = 0.01  # dummy value, not used in this simulation
amp = 100

st.sidebar.header("Parameters")
initial_tvl = st.sidebar.number_input(
    "Initial stableswap TVL (USDT + Hollar)",
    min_value=100000, max_value=10000000, value=2000000, step=100000, key="init_stableswap_tvl"
)

init_price = st.sidebar.number_input(
    "Initial price of Hollar in USDT",
    min_value=0.001, max_value=1.0, value=0.5, step=0.001, key="init_price", format="%.3f"
)
hol_pct = balance_ratio_at_price(amp, init_price)
st.sidebar.markdown("---")
hours = st.sidebar.number_input(
    "Hours to simulate",
    min_value=1, max_value=10000, value=100, step=1, key="hours"
)
blocks = hours * 300
st.sidebar.markdown("---")
buyback_speed = st.sidebar.number_input(
    "Buyback speed",
    min_value=0.00001, max_value=0.001, value=0.0001, step = 0.00001, key="buyback_speed", format="%.5f"
)
st.sidebar.text("Buyback speed is the percentage of the pool imbalance that can be corrected by HSM in one block.")
st.sidebar.markdown("---")
max_buy_price_coef = st.sidebar.number_input(
    "Max buy price coefficient",
    min_value=0.9, max_value=1.0, value=0.999, step=0.0001, key="max_buy_price_coef", format="%.4f"
)
st.sidebar.text("The max buy price coefficient times the peg is the highest price at which the HSM will buy Hollar.")
st.sidebar.markdown("---")
buy_fee = st.sidebar.number_input(
    "Buy fee", min_value=0.0001, max_value=0.01, value=0.0001, step=0.0001, key="buy_fee", format="%.4f"
)
st.sidebar.text("The buy fee is the fee spread left for arbitragers to profit from, between the stableswap price and the HSM price.")

init_hollar = initial_tvl * hol_pct
tokens = {'HOLLAR': init_hollar, 'USDT': initial_tvl * (1 - hol_pct)}
pool = StableSwapPoolState(tokens=tokens, amplification=amp, trade_fee=0.0004)
hsm = StabilityModule(
    liquidity = hsm_liquidity,
    buyback_speed = buyback_speed,
    pools = [pool],
    sell_price_fee = sell_price_fee,
    max_buy_price_coef = max_buy_price_coef,
    buy_fee = buy_fee
)
agent = Agent(holdings={'HOLLAR': 0, 'USDT': 0})

buy_amts = []
buy_prices = []
buy_spots = []
hsm_liquidity_history = []
profits = []
for i in range(blocks):
    buy_spots.append(pool.buy_spot(tkn_buy='HOLLAR', tkn_sell='USDT'))
    buy_amt, buy_price = hsm.get_buy_params(tkn='USDT')
    sell_amt = pool.calculate_sell_from_buy(tkn_buy='HOLLAR', tkn_sell='USDT', buy_quantity=buy_amt)
    hsm.arb(agent, tkn='USDT')
    buy_amts.append(buy_amt)
    buy_prices.append(buy_price)
    hsm_liquidity_history.append(hsm.liquidity['USDT'])
    hsm.update()
    profits.append(agent.holdings['USDT'])

hollar_burned = init_hollar - pool.liquidity['HOLLAR']

# Plot 1: max buy amounts from stability module
fig1, ax1 = plt.subplots()
ax1.plot(buy_amts)
ax1.set_title("max buy amounts from stability module")
ax1.set_xlabel("blocks")
ax1.set_ylabel("Hollar")
st.pyplot(fig1)

# Plot 2: stability module buy prices
fig2, ax2 = plt.subplots()
ax2.plot(buy_prices, label='HSM prices')
ax2.plot(buy_spots, label='pool prices', linestyle='--')
ax2.set_title("Prices")
ax2.set_xlabel("blocks")
ax2.legend()
st.pyplot(fig2)

# Plot 3: price difference between HSM and pool
fig3, ax3 = plt.subplots()
price_ratios = [(buy_price / buy_spot) - 1 for buy_price, buy_spot in zip(buy_prices, buy_spots) if buy_price > 0]
ax3.plot(price_ratios)
ax3.set_xlabel("blocks")
ax3.set_title("HSM buy price / pool spot price - 1")
st.pyplot(fig3)

# Plot 4: stability module USDT liquidity
fig4, ax4 = plt.subplots()
ax4.plot(hsm_liquidity_history)
ax4.set_xlabel("blocks")
ax4.set_title("stability module USDT liquidity")
st.pyplot(fig4)

# Plot 5: buy spot price vs USDT spent
usdt_spent = [1000000 - hsm_liquidity for hsm_liquidity in hsm_liquidity_history]
fig5, ax5 = plt.subplots()
ax5.plot(usdt_spent, buy_spots)
ax5.set_title("buy spot price vs USDT spent")
ax5.set_xlabel("USDT spent")
st.pyplot(fig5)

# Plot 6: profits
fig6, ax6 = plt.subplots()
ax6.plot(profits)
ax6.set_title("arbitrager profits")
ax6.set_xlabel("blocks")
ax6.set_ylabel("USDT")
st.pyplot(fig6)
