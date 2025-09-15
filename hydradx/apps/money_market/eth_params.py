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
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.trade_strategies import liquidate_cdps
from hydradx.model.processing import get_current_money_market, save_money_market, load_money_market
from hydradx.model.amm.agents import Agent
from hydradx.model.run import run
from hydradx.model.indexer_utils import get_current_omnipool_router
from hydradx.apps.display_utils import get_distribution, one_line_markdown
from hydradx.model.amm.fixed_price import FixedPriceExchange

st.markdown("""
    <style>
        .stNumberInput button {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(layout="wide")

@st.cache_data(ttl=3600, show_spinner="Loading Omnipool data (cached for 1 hour)...")
def load_omnipool_router() -> tuple[OmnipoolRouter, str]:
    block_number = 9090000
    # Add timestamp to verify caching
    import datetime
    cache_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"Cache miss! Loading omnipool at {cache_time}")
    load_router = get_current_omnipool_router(block_number)
    load_omnipool = load_router.exchanges['omnipool']
    if block_number is None:
        block_number = load_omnipool.time_step
    stableswap_pools = [pool for pool in load_router.exchanges.values() if isinstance(pool, StableSwapPoolState)]
    usd_price_lrna = (
        1 / load_omnipool.lrna_price('2-Pool-Stbl') / 1.01  # fudging this because I can't get the stableswap pool shares
    )
    load_omnipool.add_token(
        'USD', liquidity = usd_price_lrna, lrna=1
    ).stablecoin = 'USD'

    print("Loading money market data...")
    try:
        # if '/' in __file__:
        #     os.chdir(__file__[:__file__.rfind('/')])
        print(f"attempting to load money market from: {os.getcwd()}")
        load_mm = load_money_market()
    except FileNotFoundError:
        print('No local money market save file found - downloading from chain...')
        load_mm = get_current_money_market()
        load_mm.time_step = block_number

    if load_mm is None:
        print('Money market could not be loaded - check internet connection.')
        quit()

    # toss out any existing toxic CDPs
    load_mm.cdps = [cdp for cdp in load_mm.cdps if not load_mm.is_liquidatable(cdp)]
    load_mm.borrowed = {tkn: sum([cdp.debt[tkn] for cdp in load_mm.cdps if tkn in cdp.debt]) for tkn in load_mm.borrowed}

    try:
        save_money_market(load_mm, filename=f"money_market_savefile_{load_mm.time_step}")
    except FileNotFoundError:
        pass

    print("Finished downloading data.")
    return OmnipoolRouter(exchanges=[load_omnipool, load_mm, *stableswap_pools], unique_id='router'), cache_time


trade_agent = Agent(enforce_holdings=False)
def update_prices(state: GlobalState):
    for price_tkn in price_paths:
        relevant_pools = [pool for pool in state.pools.values() if price_tkn in pool.asset_list]
        for pool in relevant_pools:
            if isinstance(pool, FixedPriceExchange):
                pool.prices[price_tkn] = price_paths[price_tkn][state.time_step - 1]
            elif isinstance(pool, MoneyMarket):
                pool.assets[price_tkn].price = price_paths[price_tkn][state.time_step - 1]
                pool.prices[price_tkn] = price_paths[price_tkn][state.time_step - 1]
            elif isinstance(pool, OmnipoolState):
                # execute trades to move price towards target
                target_price = price_paths[price_tkn][state.time_step - 1] * pool.lrna_price('USD')
                pool.trade_to_price(trade_agent, tkn=price_tkn, target_price=target_price)

        state.pools['money_market'].prices[price_tkn] = price_paths[price_tkn][state.time_step - 1]
        state.external_market[price_tkn] = price_paths[price_tkn][state.time_step - 1]
        # state.pools['binance'].prices[tkn] = prices[tkn][state.time_step - 1]


router, cache_timestamp = load_omnipool_router()
omnipool = router.exchanges['omnipool']
mm = router.exchanges['money_market']

initial_cdps = [cdp.copy() for cdp in mm.cdps]
initial_cdps = sorted(initial_cdps, key=lambda cdp: mm.value_assets(cdp.collateral))[::-1]
mm.cdps = initial_cdps


@st.fragment
def plot_health_factor_distribution(money_market: MoneyMarket):
    with st.expander("Initial CDP Health Factor Distribution"):
        cdps = money_market.cdps
        resolution = st.session_state.get("resolution", 500)
        smoothing = st.session_state.get("smoothing", 3.0)
        bins, dist = get_distribution(
            [cdp.health_factor for cdp in cdps],
            [money_market.value_assets(cdp.collateral) for cdp in cdps],
            resolution=resolution,
            minimum=1,
            maximum=2,
            smoothing=smoothing
        )

        # Plot
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(bins, dist, label="Health Factor Distribution")
        ax.set_xlabel("Health Factor")
        ax.set_ylabel("Collateral-weighted density")
        st.pyplot(fig)

        resolution = st.number_input(
            label="Resolution", min_value=10, max_value=500, value=resolution, step=1, key="resolution"
        )
        smoothing = st.slider(
            label="Smoothing", min_value=0.0, max_value=10.0, value=smoothing, step=0.1, key="smoothing"
        )

plot_health_factor_distribution(mm)

with st.expander(f"Initial debt and collateral token amounts"):
    (token_name, collateral_amt, debt_amt) = st.columns(3)
    token_name.markdown("**Token**")
    collateral_amt.markdown("**Total Collateral**")
    debt_amt.markdown("**Total Debt**")
    debt_totals = {tkn: sum([cdp.debt[tkn] for cdp in mm.cdps if tkn in cdp.debt]) for tkn in mm.borrowed}
    collateral_totals = {tkn: sum([cdp.collateral[tkn] for cdp in mm.cdps if tkn in cdp.collateral]) for tkn in mm.asset_list}
    for tkn in sorted(mm.asset_list):
        with token_name:
            one_line_markdown(f"**{tkn}**")
            one_line_markdown("-------------------")
        with debt_amt:
            one_line_markdown(
                f"{debt_totals[tkn]:,.2f} ({debt_totals[tkn] * mm.assets[tkn].price:,.2f} USD)"
            )
            one_line_markdown("-------------------")

        with collateral_amt:
            one_line_markdown(
                f"{collateral_totals[tkn]:,.2f} ({collateral_totals[tkn] * mm.assets[tkn].price:,.2f} USD)"
            )
            one_line_markdown("-------------------")

# create copies of mm and omnipool to run the simulation on so we don't overwrite the cached data
mm_sim = mm.copy()
omnipool_sim = copy.deepcopy(omnipool)

# update risk parameters
mm_sim.assets['ETH'].liquidation_threshold = 0.85
mm_sim.assets['2-Pool-GETH'].liquidation_threshold = 0.75
mm_sim.assets['DOT'].liquidation_threshold = 0.85
mm_sim.assets['DOT'].supply_cap = 22_222_222
mm_sim.assets['ETH'].supply_cap = 4_444
mm_sim.assets['2-Pool-GETH'].supply_cap = 2_222

# for tkn in ['ETH', 'DOT', '2-Pool-GETH']:
#     supply_available = mm_sim.assets[tkn].supply_cap - sum(
#         [cdp.collateral[tkn] if tkn in cdp.collateral else 0 for cdp in mm_sim.cdps]
#     )
#     if supply_available > 0:
#         # assume worst-case scenario, supply cap is maxed and one huge position is on the verge of liquidation
#         new_cdp = CDP(
#             collateral={tkn: supply_available},
#             debt={
#                 'USDT': mm_sim.assets[tkn].liquidation_threshold * supply_available
#                         * mm_sim.assets[tkn].price / mm_sim.assets['USDT'].price - 1
#             }
#         )
#         new_cdp.health_factor = mm_sim.get_health_factor(new_cdp)
#         new_cdp.liquidation_threshold = mm_sim.cdp_liquidation_threshold(new_cdp)
#         mm_sim.cdps.append(new_cdp)
#         mm_sim.borrowed[tkn] += supply_available

stableswaps = [exchange for exchange in router.exchanges.values() if isinstance(exchange, StableSwapPoolState)]
stableswap_sims = [exchange.copy() for exchange in stableswaps]
router_sim = router.copy()
router_sim.exchanges = {pool.unique_id: pool for pool in [omnipool_sim, mm_sim, *stableswap_sims]}
exchanges = [router_sim, mm_sim, omnipool_sim, *stableswap_sims]

st.sidebar.info(f"Data loaded at: {cache_timestamp}")

equivalency_map = {
    'Wrapped staked ETH': 'ETH',
    'Wrapped ETH (Moonbeam Wormhole)': 'ETH',
    'aETH': 'ETH',
    'aDOT': 'DOT',
    '2-Pool-GDOT': 'DOT',
    'WETH': 'ETH',
    '2-Pool-GETH': 'ETH',
    '2-Pool-WETH': 'ETH',
    'Tether (Ethereum native)': 'USD',
    'Tether (Moonbeam Wormhole)': 'USD',
    'Tether': 'USD',
    'USDC (Ethereum native)': 'USD',
    'USDC (Moonbeam Wormhole)': 'USD',
    'DAI (Moonbeam Wormhole)': 'USD',
    'aUSDT': 'USD',
    'USDC': 'USD',
    'USDT': 'USD',
    '2-Pool-Stbl': 'USD',
    '3-Pool': 'USD',
    '4-Pool': 'USD',
    'WBTC': 'BTC',
    'interBTC': 'BTC',
    'Wrapped BTC (Moonbeam Wormhole)': 'BTC',
    'tBTC': 'BTC',
    '2-Pool': 'BTC',
}
main_tokens = [
    tkn for tkn in set(router.asset_list + list(equivalency_map.values()))
    if tkn not in equivalency_map.keys()
]
print(f"Main tokens: {main_tokens}")
price_change_defaults = {
    tkn: 0 for tkn in main_tokens
}
price_change_defaults.update({
    'ETH': -50,
})
# determine start prices for all tokens
start_price = {
    tkn: mm_sim.price(tkn) if tkn in mm.asset_list
    else omnipool.usd_price(tkn) if tkn in omnipool.asset_list
    else (
        omnipool.usd_price(equivalency_map[tkn]) if equivalency_map[tkn] in omnipool.asset_list
        else mm_sim.price(equivalency_map[tkn]) if equivalency_map[tkn] in mm_sim.asset_list else 0
    ) if tkn in equivalency_map else 0  # we'll find it in the next step
    for tkn in sorted(router.asset_list, key=lambda x: x in mm_sim.asset_list)
}

for exchange in stableswap_sims:
    priced_tokens = [tkn for tkn in exchange.asset_list if start_price[tkn] != 0]
    if priced_tokens == []:
        for tkn in exchange.asset_list:
            start_price[tkn] = 1  # default price for USD
    else:
        for tkn in exchange.asset_list:
            if tkn not in priced_tokens:
                start_price[tkn] = exchange.price(tkn, start_price[priced_tokens[0]]) * start_price[priced_tokens[0]]
    # # now that we know what the assets are worth, we can calculate shares
    # if exchange.unique_id in omnipool.asset_list:
    #     exchange.shares = omnipool.liquidity[exchange.unique_id]
    # elif exchange.unique_id in mm.asset_list:
    #     exchange.shares = sum([exchange.liquidity[tkn] * start_price[tkn] for tkn in exchange.asset_list]) / mm.price(exchange.unique_id)
    # else:
    #     exchange.shares = sum([start_price[tkn] * exchange.liquidity[tkn] for tkn in exchange.asset_list])
    #     # exchanges.remove(exchange)

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
        ) for tkn in sorted(main_tokens, key=lambda x: router.liquidity[x] if x in router.liquidity else float('inf'), reverse=True)
    }
    # update price change defaults for equivalent tokens
    price_factor.update({
        tkn: price_factor[equivalency_map[tkn]] for tkn in equivalency_map.keys()
        if equivalency_map[tkn] in main_tokens
    })

# calculate price path from start price to final price
all_price_change_tokens = [tkn for tkn in price_factor if price_factor[tkn] != 0]
final_price = {tkn: start_price[tkn] * (1 + price_factor[tkn] / 100) for tkn in start_price}
price_paths = {
    tkn: [
        start_price[tkn]] + [start_price[tkn] - (start_price[tkn] - final_price[tkn]) * ((i + 1) / time_steps)
        for i in range(time_steps)
    ] + [final_price[tkn]] for tkn in start_price
}
time_steps += 1  # to account for final, stable price step
#
# config_list = [
#     {'exchanges': {'router': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
#     {'exchanges': {'omnipool': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
# ]

initial_state = GlobalState(
    pools=exchanges,
    agents={
        'liquidator': Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('router')),
        # 'arbitrageur': Agent(
        #     enforce_holdings=False,
        #     trade_strategy=general_arbitrage(
        #         exchanges=[omnipool_sim, mm_sim, *stableswap_sims],
        #         # config=config_list,
        #         equivalency_map=equivalency_map
        #     )
        # )
    },
    evolve_function=update_prices,
    external_market=start_price
)

st.header("Running simulation...")
with st.spinner(f"Running {time_steps} simulation steps..."):
    sim_start = time.time()
    events = run(initial_state, time_steps=time_steps, silent=True)

    sim_time = time.time() - sim_start
    st.sidebar.info(f"Simulation completed in {sim_time:.2f} seconds")

final_mm = events[-1].pools['money_market']
for cdp in final_mm.cdps:
    cdp.health_factor = final_mm.get_health_factor(cdp)

with st.expander(f"Prices over {time_steps} steps"):
    for tkn in sorted(start_price, key=lambda x: equivalency_map[x] if x in equivalency_map else x):
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlabel("Time Steps")
        ax.set_ylabel(f"{tkn} price (USD)")
        tkn_price_path = [
            event.pools['omnipool'].usd_price(tkn)
            if tkn in event.pools['omnipool'].asset_list
            else event.pools['money_market'].price(tkn)
            for event in events
        ]
        if max(tkn_price_path) == 0:
            continue
        if abs(min(tkn_price_path) / max(tkn_price_path) - 1) < 0.01:
            continue
        # if abs(1 - tkn_price_path[-1] / final_price[tkn]) > 0.01:
        #     st.warning(f"{tkn} price did not reach expected final price of {final_price[tkn]} USD")
        ax.plot(tkn_price_path, label=f"{tkn} price")
        ax.annotate(
            f"${tkn_price_path[0]:.2f}",
            xy=(0, tkn_price_path[0]),
            xytext=(0, tkn_price_path[0] * (1.05 if tkn_price_path[0] < tkn_price_path[-1] else 0.95))
        )
        ax.scatter(marker='o', s=20, x=0, y=tkn_price_path[0], color='#009')
        ax.annotate(
            f"${tkn_price_path[-1]:.2f}",
            xy=(len(tkn_price_path)-1, tkn_price_path[-1]),
            xytext=(len(tkn_price_path) - 1.4, tkn_price_path[-1] * (1.05 if tkn_price_path[0] > tkn_price_path[-1] else 0.95))
        )
        ax.scatter(marker='o', s=20, x=len(tkn_price_path)-1, y=tkn_price_path[-1], color='#009')
        st.pyplot(fig)

with st.expander(f"Liquidator holdings"):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(f"Liquidator Holdings (normalized to initial USD price)")
    for tkn in events[-1].agents['liquidator'].holdings:
        liquidator_holdings = [
            event.agents['liquidator'].get_holdings(tkn) * start_price[tkn] / start_price['USD']
            for event in events
        ]
        if max(liquidator_holdings) == 0:
            continue
        ax.plot(liquidator_holdings, label=f"{tkn}")
    ax.legend()
    st.pyplot(fig)


@st.fragment
def plot_toxic_debt():
    with (st.expander(f"Toxic debt")):
        st.radio(
            label="View toxic debt as:",
            options=["absolute (USD)", "percentage", "USD at initial prices"],
            index=1,
            key="toxic_debt_view"
        )
        st.checkbox(label="breakdown by debt token", key="toxic_debt_breakdown", value=True)
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Toxic Debt")
        toxic_debt = {
            tkn: [
                sum([
                    cdp.debt[tkn] if event.pools['money_market'].is_toxic(cdp) and tkn in cdp.debt else 0
                    for cdp in event.pools['money_market'].cdps
                ])
                for event in events
            ] for tkn in mm.asset_list
        }
        toxic_debt.update(
            {"all": [
                sum([
                    toxic_debt[tkn][i] * (
                        events[i].pools['money_market'].assets[tkn].price
                        if st.session_state["toxic_debt_view"] != "USD at initial prices"
                        else start_price[tkn]
                    ) for tkn in toxic_debt
                ]) for i in range(time_steps)
            ]}
        )
        if not st.session_state.toxic_debt_breakdown:
            toxic_debt = {'all': toxic_debt['all']}

        if max(toxic_debt.values(), key=lambda value: max(value)) == 0:
            st.info("No toxic debt at any point in the simulation.")
            return

        money_market_values = [
            sum([
                event.pools['money_market'].value_assets(cdp.debt)
                for cdp in event.pools['money_market'].cdps
            ]) for event in events
        ]

        for i, (tkn, debt) in enumerate(toxic_debt.items()):
            if max(debt) == 0:
                continue
            if st.session_state.toxic_debt_view == "percentage":
                ax.plot(
                    [
                        debt[i] * (event.pools['money_market'].assets[tkn].price if tkn in mm.asset_list else 1)
                        / money_market_values[i] * 100
                        for i, event in enumerate(events)
                    ], label=f"{tkn}" if st.session_state.toxic_debt_breakdown else None
                )
                ax.set_title("Toxic debt as a percentage of total debt")
            elif st.session_state.toxic_debt_view == "USD at initial prices":
                ax.plot(
                    [debt[i] * (start_price[tkn] if tkn in start_price else 1) for i in range(len(debt))],
                    label=f"{tkn}" if st.session_state.toxic_debt_breakdown else None
                )
                ax.set_title("Toxic debt in USD valued at initial prices")
            else:
                ax.plot(
                    [
                        debt[i] * (event.pools['money_market'].assets[tkn].price if tkn in mm.asset_list else 1)
                        for i, event in enumerate(events)
                    ], label=f"{tkn}" if st.session_state.toxic_debt_breakdown else None
                )
                ax.set_title("Toxic debt in USD")
        ax.legend()
        st.pyplot(fig)

plot_toxic_debt()


@st.fragment
def plot_collateral():
    with (st.expander(f"Collateral")):
        st.radio(
            label="View collateral as:",
            options=["absolute USD", "USD at initial prices"],
            index=1,
            key="collateral_view"
        )
        st.checkbox(label="breakdown by token", key="collateral_breakdown", value=True)
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Toxic Debt")
        collateral_tokens = {
            tkn: [
                sum([
                    cdp.collateral[tkn] if tkn in cdp.collateral else 0
                    for cdp in event.pools['money_market'].cdps
                ])
                for event in events
            ] for tkn in mm.asset_list
        }
        collateral_tokens.update(
            {"all": [
                sum([
                    collateral_tokens[tkn][i] * (
                        events[i].pools['money_market'].assets[tkn].price
                        if st.session_state.collateral_view != "USD at initial prices"
                        else start_price[tkn]
                    ) for tkn in collateral_tokens
                ])
                for i in range(time_steps)
            ]}
        )
        if not st.session_state.collateral_breakdown:
            collateral_tokens = {'all': collateral_tokens['all']}

        for i, (tkn, collateral) in enumerate(collateral_tokens.items()):
            if max(collateral) == 0:
                continue
            if st.session_state.collateral_view == "USD at initial prices":
                ax.plot(
                    [
                        collateral[i] * (start_price[tkn] if tkn in start_price else 1) for i in range(time_steps)
                    ], label=f"{tkn}" if st.session_state.collateral_breakdown else None
                )
                ax.set_title("Collateral in USD valued at initial prices")
            else:
                ax.plot(
                    [
                        collateral[i] * (event.pools['money_market'].assets[tkn].price if tkn in mm.asset_list else 1)
                        for i, event in enumerate(events)
                    ], label=f"{tkn}" if st.session_state.collateral_breakdown else None
                )
                ax.set_title("Collateral in USD")
        ax.legend()
        st.pyplot(fig)

plot_collateral()