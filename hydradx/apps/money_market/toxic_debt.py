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
from hydradx.model.amm.trade_strategies import liquidate_cdps, TradeStrategy, general_arbitrage
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
    target_price_lrna = target_price * pool.lrna_price('USDT')
    k = pool.lrna[tkn_sell] * pool.liquidity[tkn_sell]
    print(target_price_lrna)
    target_x = math.sqrt(k / target_price_lrna)
    dx = target_x - pool.liquidity[tkn_sell]
    return dx


def update_prices(state):
    prices = {
        tkn: state.pools['omnipool'].usd_price(tkn, 'USDT')
        for tkn in list(set(state.pools['money_market'].prices) & set(omnipool.asset_list))
    }
    state.pools['money_market'].prices.update(prices)

def schedule_swaps(swaps: list[list[dict]]):
    class Strategy:
        def __init__(self, pool_id: str):
            self.initial_time_step = -1
            self.pool_id = pool_id
        def execute(self, state: GlobalState, agent_id: str):
            if self.initial_time_step == -1:
                self.initial_time_step = state.time_step
            agent = state.agents[agent_id]
            for trade in swaps [state.time_step - self.initial_time_step]:
                state.pools[self.pool_id].swap(
                    tkn_sell=trade['tkn_sell'],
                    tkn_buy=trade['tkn_buy'],
                    sell_quantity=trade['sell_quantity'] if 'sell_quantity' in trade else 0,
                    buy_quantity=trade['buy_quantity'] if 'buy_quantity' in trade else 0,
                    agent=agent
                )

            return state

    return TradeStrategy(Strategy('omnipool').execute, name="scheduled_swaps")

router, cache_timestamp = load_omnipool_router()
omnipool, mm = router.exchanges.values()
st.sidebar.info(f"Data loaded at: {cache_timestamp}")

price_change_defaults = {
    tkn: 0 for tkn in mm.asset_list
}
price_change_defaults.update({
    'DOT': -75,
})
with st.sidebar:
    time_steps = st.number_input(
        label="Time Steps",
        min_value=1,
        max_value=100,
        value=10
    )
    price_factor = {
        tkn: st.number_input(
            label=f"{tkn} price change: ",
            min_value=-99,
            max_value=1000,
            value=price_change_defaults[tkn],
            key=tkn
        ) for tkn in mm.asset_list
    }

start_price = {tkn: omnipool.price(tkn, 'USDT') for tkn in mm.asset_list}

# calculate price path from start price to crash price
final_price = {tkn: start_price[tkn] * (1 + price_factor[tkn] / 100) for tkn in mm.asset_list}
prices = {
    tkn: [
        start_price[tkn]] + [start_price[tkn] - (start_price[tkn] - final_price[tkn]) * ((i + 1) / time_steps)
        for i in range(time_steps)
    ] for tkn in mm.asset_list
}
full_trades = {tkn: [trade_to_price(omnipool, tkn, price) for price in prices[tkn]] for tkn in mm.asset_list}
trade_sequence = [
    [
        {
            'tkn_sell': tkn,
            'tkn_buy': 'LRNA',
            'sell_quantity': full_trades[tkn][i] - full_trades[tkn][i-1]
        } if full_trades[tkn][i] != full_trades[tkn][i-1] else {
            'tkn_sell': 'LRNA',
            'tkn_buy': tkn,
            'buy_quantity': full_trades[tkn][i] - full_trades[tkn][i - 1]
        } for tkn in mm.asset_list
    ] for i in range(1, time_steps + 1)
]
omnipool_sim = omnipool.copy()
mm_sim = mm.copy()
initial_state = GlobalState(
    pools=[mm_sim, omnipool_sim],
    agents={
        'liquidator': Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool')),
        'panic seller': Agent(
            enforce_holdings=False,
            trade_strategy=schedule_swaps(trade_sequence)
        ),
        'arbitrageur': Agent(
            enforce_holdings=False,
            trade_strategy=general_arbitrage(
                exchanges=[mm_sim, omnipool_sim]
            )
        )
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

for tkn in mm.asset_list:
    fig, ax = plt.subplots()
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(f"{tkn} price (USD)")
    ax.plot([
        event.pools['omnipool'].price(tkn, 'USDT') for event in events
    ])
    st.pyplot(fig)
