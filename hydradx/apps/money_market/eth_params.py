import copy

from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.money_market import MoneyMarket, MoneyMarketAsset, CDP
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.trade_strategies import liquidate_cdps, general_arbitrage
from hydradx.model.processing import get_current_money_market, save_money_market, load_money_market
from hydradx.model.amm.agents import Agent
from hydradx.model.run import run
from hydradx.model.amm.trade_strategies import schedule_swaps
from hydradx.model.indexer_utils import get_current_omnipool_router

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner="Loading Omnipool data (cached for 1 hour)...")
def load_omnipool_router() -> tuple[OmnipoolRouter, str]:
    block_number = None  # 8450000
    # Add timestamp to verify caching
    import datetime
    cache_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Cache miss! Loading omnipool at {cache_time}")
    load_router = get_current_omnipool_router(block_number)
    load_omnipool = load_router.exchanges['omnipool']
    stableswap_pools = [pool for pool in load_router.exchanges.values() if isinstance(pool, StableSwapPoolState)]
    usd_price_lrna = (
        1 / load_omnipool.lrna_price('2-Pool-Stbl') / 1.01  # fudging this because I can't get the stableswap pool shares
    )
    load_omnipool.add_token(
        'USD', liquidity = usd_price_lrna, lrna=1
    ).stablecoin = 'USD'

    print("Loading money market data...")
    try:
        load_mm = load_money_market(filename=f"money_market_at_{block_number}")
    except FileNotFoundError:
        load_mm = get_current_money_market()

    if load_mm is None:
        print('Money market could not be loaded - check internet connection.')
        quit()

    # update risk parameters
    load_mm.assets['ETH'].liquidation_threshold = 0.85
    load_mm.assets['2-Pool-GETH'].liquidation_threshold = 0.75
    load_mm.assets['DOT'].liquidation_threshold = 0.85
    load_mm.assets['DOT'].supply_cap = 22_222_222
    load_mm.assets['ETH'].supply_cap = 4_444
    load_mm.assets['2-Pool-GETH'].supply_cap = 2_222

    try:
        save_money_market(load_mm, filename=f"money_market_at_{block_number}")
    except FileNotFoundError:
        pass

    for tkn in ['ETH', 'DOT', '2-Pool-GETH']:
        supply_available = load_mm.assets[tkn].supply_cap - sum(
            [cdp.collateral[tkn] if tkn in cdp.collateral else 0 for cdp in load_mm.cdps]
        )
        if supply_available > 0:
            # assume worst-case scenario, supply cap is maxed and one huge position is on the verge of liquidation
            new_cdp = CDP(
                collateral={tkn: supply_available},
                debt={'USDT': load_mm.assets[tkn].liquidation_threshold * supply_available * load_mm.assets[tkn].price - 1}
            )
            load_mm.cdps.append(new_cdp)
            load_mm.borrowed[tkn] += supply_available

    print("Finished downloading data.")
    return OmnipoolRouter(exchanges=[load_omnipool, load_mm, *stableswap_pools], unique_id='router'), cache_time


def trade_to_price(pool, tkn_sell, target_price):
    if tkn_sell not in pool.liquidity:
        return 0
    # this is the target price in USD - convert to LRNA
    target_price_lrna = target_price * pool.lrna_price('USD')
    k = pool.lrna[tkn_sell] * pool.liquidity[tkn_sell]
    print(target_price_lrna)
    target_x = math.sqrt(k / target_price_lrna)
    dx = target_x - pool.liquidity[tkn_sell]
    return dx


def update_prices(state: GlobalState):
    for tkn in prices:
        state.pools['money_market'].prices[tkn] = prices[tkn][state.time_step - 1]
        state.external_market[tkn] = prices[tkn][state.time_step - 1]
        # state.pools['binance'].prices[tkn] = prices[tkn][state.time_step - 1]

router, cache_timestamp = load_omnipool_router()
omnipool = router.exchanges['omnipool']
mm = router.exchanges['money_market']
stableswaps = [exchange for exchange in router.exchanges.values() if isinstance(exchange, StableSwapPoolState)]
st.sidebar.info(f"Data loaded at: {cache_timestamp}")

price_change_defaults = {
    tkn: 0 for tkn in router.asset_list
}
price_change_defaults.update({
    'ETH': -75
})
equivalency_map = {
    'interBTC': 'tBTC',
    'Wrapped staked ETH': 'ETH',
    'Wrapped ETH (Moonbeam Wormhole)': 'ETH',
    'aETH': 'ETH',
    'aDOT': 'DOT',
    'WETH': 'ETH',
    'Tether (Ethereum native)': 'USD',
    'Tether (Moonbeam Wormhole)': 'USD',
    'Tether': 'USD',
    'USDC (Ethereum native)': 'USD',
    'USDC (Moonbeam Wormhole)': 'USD',
    'DAI (Moonbeam Wormhole)': 'USD',
    'aUSDT': 'USD',
    'USDC': 'USD',
    'USDT': 'USD',
    'aUSDT': 'USD',
    'WBTC': 'tBTC',
    'interBTC': 'tBTC',
    'Wrapped BTC (Moonbeam Wormhole)': 'tBTC',
}
# update price change defaults for equivalent tokens
price_change_defaults.update({
    tkn: price_change_defaults[equivalency_map[tkn]] for tkn in equivalency_map.keys()
    if equivalency_map[tkn] in price_change_defaults
})
start_price = {
    tkn: omnipool.usd_price(tkn) if tkn in omnipool.asset_list
    else mm.price(tkn) if tkn in mm.asset_list
    else 0  # default price for USD
    for tkn in sorted(router.asset_list, key=lambda x: x in mm.asset_list)
}
for exchange in stableswaps:
    priced_tokens = [tkn for tkn in exchange.asset_list if start_price[tkn] != 0]
    if priced_tokens == []:
        for tkn in exchange.asset_list:
            start_price[tkn] = 1  # default price for USD
    else:
        for tkn in exchange.asset_list:
            if tkn not in priced_tokens:
                start_price[tkn] = exchange.price(tkn, start_price[priced_tokens[0]]) * start_price[priced_tokens[0]]

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
        ) for tkn in start_price
    }

# calculate price path from start price to final price
all_price_change_tokens = [tkn for tkn in price_factor if price_factor[tkn] != 0]
final_price = {tkn: start_price[tkn] * (1 + price_factor[tkn] / 100) for tkn in start_price}
prices = {
    tkn: [
        start_price[tkn]] + [start_price[tkn] - (start_price[tkn] - final_price[tkn]) * ((i + 1) / time_steps)
        for i in range(time_steps)
    ] for tkn in start_price
}
full_trades = {tkn: [trade_to_price(omnipool, tkn, price) for price in prices[tkn]] for tkn in start_price}
# create trade sequence for omnipool assets whose price changes
omnipool_sell_assets = [tkn for tkn in list(set(final_price.keys()) & set(omnipool.asset_list)) if final_price[tkn] != start_price[tkn]]
trade_sequence = [
    [
        {
            'tkn_sell': tkn,
            'tkn_buy': 'LRNA',
            'sell_quantity': full_trades[tkn][i] - full_trades[tkn][i-1]
        } if full_trades[tkn][i] < full_trades[tkn][i-1] else {
            'tkn_sell': 'LRNA',
            'tkn_buy': tkn,
            'buy_quantity': full_trades[tkn][i - 1] - full_trades[tkn][i]
        } for tkn in omnipool_sell_assets
    ] for i in range(1, time_steps + 1)
]
trade_sequence.extend([[
    {'tkn_sell': tkn, 'tkn_buy': 'LRNA', 'sell_quantity': 0}
    for tkn in omnipool_sell_assets
]])
time_steps = len(trade_sequence)
omnipool_sim = copy.deepcopy(omnipool)
mm_sim = mm.copy()

# config_list = [
#     {'exchanges': {'omnipool': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
#     {'exchanges': {'omnipool': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
# ]

initial_state = GlobalState(
    pools=[router, mm_sim, omnipool_sim, *stableswaps],
    agents={
        'liquidator': Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool')),
        'panic seller': Agent(
            enforce_holdings=False,
            trade_strategy=schedule_swaps('omnipool', trade_sequence)
        ),
        # 'arbitrageur': Agent(
        #     enforce_holdings=False,
        #     trade_strategy=general_arbitrage(
        #         exchanges=list(router.exchanges.values()),
        #         # config=config_list,
        #         equivalency_map=equivalency_map
        #     )
        # )
    },
    evolve_function=update_prices,
    external_market=start_price
)

with st.spinner(f"Running {time_steps} simulation steps..."):
    sim_start = time.time()
    events = []
    for time_step in range(time_steps):
        events.extend(run(initial_state, time_steps, silent=True))
        for cdp in mm.cdps:
            if cdp.health_factor < 1:
                print(f"CDP {cdp.unique_id} is not liquidated at time step {time_step + 1}")
                # mm_sim.liquidate(cdp, agent=initial_state.agents['liquidator'])
        for tkn in list(set(prices.keys()) & set(mm_sim.asset_list)):
            mm_sim.prices[tkn] = prices[tkn][time_step]
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

for tkn in start_price:
    fig, ax = plt.subplots()
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(f"{tkn} price (USD)")
    tkn_price_path = [
        event.pools['omnipool'].usd_price(tkn)
        if tkn in event.pools['omnipool'].asset_list
        else event.pools['money_market'].price(tkn)
        for event in events
    ]
    if abs(1 - tkn_price_path[-1] / final_price[tkn]) > 0.01:
        st.warning(f"{tkn} price did not reach expected final price of {final_price[tkn]} USD")
    ax.plot(tkn_price_path, label=f"{tkn} price")
    st.pyplot(fig)
