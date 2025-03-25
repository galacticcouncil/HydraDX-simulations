from matplotlib import pyplot as plt
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState, balance_ratio_at_price
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule

hsm_liquidity = {'USDT': 1000000}
initial_tvl = 1000000
sell_price_fee = 0.01
buy_fee = 0.0001
amp = 100

st.sidebar.header("Parameters")
st.sidebar.markdown("*Percentage of the stableswap pool that is Hollar initiailly, in absolute token quantities*")
init_price = st.sidebar.number_input(
    "Initial price of Hollar in USDT",
    min_value=0.001, max_value=1.0, value=0.9, step=0.001, key="init_price", format="%.3f"
)
hol_pct = balance_ratio_at_price(amp, init_price)
hours = st.sidebar.number_input(
    "Hours to simulate",
    min_value=1, max_value=10000, value=100, step=1, key="hours"
)
blocks = hours * 300
st.sidebar.markdown("*Buyback speed is recipricol of buyback denominator*")
buyback_denom = st.sidebar.number_input(
    "Buyback denominator",
    min_value=10, max_value=1000000, value=10000, step = 10, key="buyback_denom"
)
buyback_speed = 1 / buyback_denom
max_buy_price_coef = st.sidebar.number_input(
    "Max buy price coefficient",
    min_value=0.9, max_value=1.0, value=0.999, step=0.0001, key="max_buy_price_coef", format="%.4f"
)

init_hollar = initial_tvl * hol_pct
tokens = {'HOLLAR': init_hollar, 'USDT': initial_tvl * (1 - hol_pct)}
pool = StableSwapPoolState(tokens=tokens, amplification=amp, trade_fee=0.0001)
hsm = StabilityModule(
    liquidity = hsm_liquidity,
    buyback_speed = buyback_speed,
    pools = [pool],
    sell_price_fee = sell_price_fee,
    max_buy_price_coef = max_buy_price_coef,
    buy_fee = buy_fee
)
agent = Agent(holdings={'HOLLAR': 0, 'USDT': 1000})
print("initial pool price:")
print(pool.buy_spot(tkn_buy='HOLLAR', tkn_sell='USDT'))




buy_amts = []
buy_prices = []
buy_spots = []
hsm_liquidity_history = []
for i in range(blocks):
    buy_amt, buy_price = hsm.get_buy_params(tkn='USDT')
    hsm.arb(agent, tkn='USDT')
    buy_amts.append(buy_amt)
    buy_prices.append(buy_price)
    buy_spots.append(pool.buy_spot(tkn_buy='HOLLAR', tkn_sell='USDT'))
    hsm_liquidity_history.append(hsm.liquidity['USDT'])
    hsm.update()

hollar_burned = init_hollar - pool.liquidity['HOLLAR']

print(pool.liquidity)
print('total Hollar bought off market: ' + str(hollar_burned))
print('total USDT paid by stability module: ' + str(1000000 - hsm.liquidity['USDT']))
print('arbitrager profits: ' + str(agent.holdings['USDT']))
print('average price paid by stability module: ' + str((1000000 - hsm.liquidity['USDT']) / hollar_burned))
print("final pool price: " + str(pool.buy_spot(tkn_buy='HOLLAR', tkn_sell='USDT')))

# Plot 1: max buy amounts from stability module
fig1, ax1 = plt.subplots()
ax1.plot(buy_amts)
ax1.set_title("max buy amounts from stability module")
st.pyplot(fig1)

# Plot 2: stability module buy prices
fig2, ax2 = plt.subplots()
ax2.plot(buy_prices)
ax2.set_title("stability module buy prices")
st.pyplot(fig2)

# Plot 3: stableswap pool prices
fig3, ax3 = plt.subplots()
ax3.plot(buy_spots)
ax3.set_title("stableswap pool prices")
st.pyplot(fig3)

# Plot 4: stability module USDT liquidity
fig4, ax4 = plt.subplots()
ax4.plot(hsm_liquidity_history)
ax4.set_title("stability module USDT liquidity")
st.pyplot(fig4)

# Plot 5: buy spot price vs USDT spent
usdt_spent = [1000000 - hsm_liquidity for hsm_liquidity in hsm_liquidity_history]
fig5, ax5 = plt.subplots()
ax5.plot(usdt_spent, buy_spots)
ax5.set_title("buy spot price vs USDT spent")
st.pyplot(fig5)
