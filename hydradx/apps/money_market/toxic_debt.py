from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.global_state import GlobalState
# from hydradx.model.amm.money_market import MoneyMarket, MoneyMarketAsset
from hydradx.model.amm.omnipool_router import OmnipoolRouter
# from hydradx.model.amm.omnipool_amm import OmnipoolState, DynamicFee
from hydradx.model.amm.trade_strategies import liquidate_cdps, TradeStrategy
from hydradx.model.processing import get_current_money_market, get_stableswap_data, Pool
from hydradx.model.amm.agents import Agent
from hydradx.model.run import run
from hydradx.model.amm.trade_strategies import constant_swaps
from hydradx.model.indexer_utils import get_current_omnipool, get_asset_info

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner="Loading Omnipool data (cached for 1 hour)...")
def load_omnipool_router() -> tuple[OmnipoolRouter, str]:
    # Add timestamp to verify caching
    import datetime
    cache_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Cache miss! Loading omnipool at {cache_time}")
    omnipool = get_current_omnipool()
    mm = get_current_money_market()
    lrna_price = mm.price('DOT') / omnipool.lrna_price('DOT')

    asset_info = get_asset_info()
    stable_swaps = get_stableswap_data()

    for pool in stable_swaps.values():
        pool_name = asset_info[pool.pool_id].name
        pool_total = sum([
            int(value) / 10 ** asset_info[tkn].decimals for tkn, value in pool.reserves.items()
        ])
        for tkn in pool.assets:
            tkn_name = asset_info[tkn.asset_id].symbol
            liquidity = (
                int(pool.reserves[tkn.asset_id]) / 10 ** tkn.decimals if tkn.asset_id in pool.reserves
                else pool.shares / 10 ** tkn.decimals / len(pool.assets)
            )
            if tkn_name in mm.asset_list:
                lrna = liquidity * mm.price(tkn_name) / lrna_price
            elif tkn_name in omnipool.asset_list:
                lrna = liquidity / omnipool.lrna_price(tkn_name)
            elif pool_name in omnipool.asset_list:
                lrna = liquidity / pool_total * omnipool.lrna[pool_name]
            else:
                lrna = 0  # no data

            if tkn_name not in omnipool.asset_list:
                omnipool.add_token(
                    tkn=tkn_name,
                    liquidity=liquidity,
                    lrna=lrna
                )
            else:
                omnipool.liquidity[tkn_name] += liquidity
                omnipool.lrna[tkn_name] += lrna
                omnipool.shares[tkn_name] += liquidity
                omnipool.protocol_shares[tkn_name] += liquidity

    gigadot_info = stable_swaps[690]
    gigadot_liquidity = gigadot_info.shares / 10 ** 18
    pool_price = sum(mm.price(tkn) for tkn in ['DOT', 'vDOT']) / 2  # best guess
    omnipool.add_token(
        tkn='2-Pool-GDOT',
        liquidity=gigadot_liquidity,
        lrna=gigadot_liquidity * pool_price / lrna_price,
        shares=gigadot_liquidity,
    )
    print("Finished downloading data.")
    return OmnipoolRouter([omnipool, mm]), cache_time


def trade_to_price(pool, tkn_sell, target_price):
    # this is the target price in USD - convert to LRNA
    target_price_lrna = target_price / pool.usd_price(tkn_sell, 'USDT') * pool.lrna_price(tkn_sell)
    k = pool.lrna[tkn_sell] * pool.liquidity[tkn_sell]
    target_x = math.sqrt(k / target_price_lrna)
    dx = target_x - pool.liquidity[tkn_sell]
    return dx


def update_prices(state):
    prices = {
        tkn: state.pools['omnipool'].usd_price(tkn, 'USDT')
        for tkn in list(set(state.pools['money_market'].prices) & set(omnipool.asset_list))
    }
    state.pools['money_market'].prices.update(prices)

def schedule_swaps(tkn_sell: str, tkn_buy: str, quantity: list[float]):
    class Strategy:
        def __init__(self, pool_id: str):
            self.initial_time_step = -1
            self.pool_id = pool_id
        def execute(self, state: GlobalState, agent_id: str):
            if self.initial_time_step == -1:
                self.initial_time_step = state.time_step
            agent = state.agents[agent_id]
            state.pools[self.pool_id].swap(
                tkn_sell=tkn_sell,
                tkn_buy=tkn_buy,
                sell_quantity=quantity[state.time_step - self.initial_time_step],
                agent=agent
            )
            return state

    return TradeStrategy(Strategy('omnipool').execute, name="scheduled_swaps")


router, cache_timestamp = load_omnipool_router()
omnipool, mm = router.exchanges.values()
st.sidebar.info(f"Data loaded at: {cache_timestamp}")

with st.sidebar:
    crash_asset = st.selectbox(
        label="Crashing Asset",
        options=mm.asset_list,
        index=mm.asset_list.index('DOT') if 'DOT' in mm.asset_list else 0
    )
    crash_factor = st.number_input(
        label="Crash Factor",
        min_value=1,
        max_value=99,
        value=75
    )
    time_steps = st.number_input(
        label="Time Steps",
        min_value=1,
        max_value=100,
        value=10
    )
start_price = omnipool.price(crash_asset, 'USDT')

# calculate price path from start price to crash price
final_price = start_price * (1 - crash_factor / 100)
prices = [start_price] + [start_price - (start_price - final_price) * ((i + 1) / time_steps) for i in range(time_steps)]
full_trades = [trade_to_price(omnipool, crash_asset, price) for price in prices]
trade_sequence = [full_trades[i] - full_trades[i-1] for i in range(1, time_steps + 1)]
initial_state = GlobalState(
    pools={
        'money_market': mm.copy(),
        'omnipool': omnipool.copy()
    },
    agents={
        'liquidator': Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool')),
        'panic seller': Agent(
            enforce_holdings=False,
            trade_strategy=schedule_swaps(crash_asset, 'LRNA', trade_sequence)
        ),
    },
    evolve_function=update_prices
)

with st.spinner(f"Running {time_steps} simulation steps..."):
    sim_start = time.time()
    events = run(initial_state, time_steps=time_steps, silent=True)
    sim_time = time.time() - sim_start
    st.sidebar.info(f"Simulation completed in {sim_time:.2f} seconds")

fig1, ax1 = plt.subplots()
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Toxic Debt")
ax1.plot([
    sum([
        event.pools['money_market'].value_assets(cdp.debt)
        if event.pools['money_market'].is_toxic(cdp) else 0
        for cdp in event.pools['money_market'].cdps
    ]) / sum([
        event.pools['money_market'].value_assets(cdp.debt)
        for cdp in event.pools['money_market'].cdps
    ]) * 100 for event in events
])
ax1.set_title("Toxic debt as a percentage of total debt")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.set_xlabel("Time Steps")
ax2.set_ylabel(f"{crash_asset} price (USD)")
ax2.plot([
    event.pools['omnipool'].price(crash_asset, 'USDT') for event in events
])
st.pyplot(fig2)
