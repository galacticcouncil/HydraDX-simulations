import copy
from typing import Callable

from matplotlib import pyplot as plt
import sys, os, math
import streamlit as st, streamlit.components.v1 as components
import time

from streamlit import form_submit_button

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.global_state import GlobalState
from hydradx.model.amm.omnipool_router import OmnipoolRouter
from hydradx.model.amm.money_market import MoneyMarket, MoneyMarketAsset, CDP
from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.trade_strategies import liquidate_cdps, schedule_swaps
from hydradx.model.processing import get_current_money_market, save_money_market, load_money_market as load_money_market_from_file
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
        section[data-testid="stSidebar"] { min-width: 380px !important; width: 380px !important; }
        section[data-testid="stSidebar"] > div { width: 380px !important; }
        
        section[data-testid="stSidebar"] div.stButton > button {
          padding: 0 6px !important;
          min-height: 0 !important;
          height: 22px !important;
          line-height: 1 !important;
          font-size: 14px !important;
          border-radius: 6px !important;
        }
        
        section[data-testid="stSidebar"] div.stButton { text-align: right; }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
print("App start")

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
    print('finished downloading omnipool data')
    return OmnipoolRouter(exchanges=[load_omnipool, *stableswap_pools], unique_id='router'), cache_time


def load_money_market(block_number: int) -> MoneyMarket | None:
    print("Loading money market data...")
    try:
        # if '/' in __file__:
        #     os.chdir(__file__[:__file__.rfind('/')])
        print(f"attempting to load money market from: {os.getcwd()}")
        load_mm = load_money_market_from_file()
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
        save_money_market(load_mm, filename=f"money_market_savefile_{block_number}")
    except FileNotFoundError:
        pass

    print("Finished downloading money_market data.")
    return load_mm


initial_router, cache_timestamp = load_omnipool_router()
initial_omnipool = initial_router.exchanges['omnipool']
initial_mm = load_money_market(initial_omnipool.time_step)
initial_cdps = [cdp.copy() for cdp in initial_mm.cdps]
initial_cdps = sorted(initial_cdps, key=lambda cdp: initial_mm.value_assets(cdp.collateral))[::-1]
initial_mm.cdps = initial_cdps
initial_router.exchanges['money_market'] = initial_mm
initial_stableswaps = [exchange for exchange in initial_router.exchanges.values() if isinstance(exchange, StableSwapPoolState)]
if 'money_market' not in st.session_state:
    st.session_state['money_market'] = initial_mm.copy()
mm = st.session_state['money_market']

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
    tkn for tkn in set(initial_router.asset_list + list(equivalency_map.values()))
    if tkn not in equivalency_map.keys()
]
main_tokens = sorted(main_tokens, key=lambda x: initial_router.liquidity[x] if x in initial_router.liquidity else float('inf'), reverse=True)
print(f"Main tokens: {main_tokens}")
price_change_defaults = {
    tkn: 0 for tkn in main_tokens
}
price_change_defaults.update({
    'DOT': -50,
})
# determine start prices for all tokens
start_price = {
    tkn: initial_mm.price(tkn) if tkn in initial_mm.asset_list
    else initial_omnipool.usd_price(tkn) if tkn in initial_omnipool.asset_list
    else (
        initial_omnipool.usd_price(equivalency_map[tkn]) if equivalency_map[tkn] in initial_omnipool.asset_list
        else initial_mm.price(equivalency_map[tkn]) if equivalency_map[tkn] in initial_mm.asset_list else 0
    ) if tkn in equivalency_map
    else 0
    for tkn in sorted(set(initial_router.asset_list).union(set(equivalency_map.values())), key=lambda x: x in initial_mm.asset_list)
}
for tkn in start_price:
    if start_price[tkn] == 0:
        eqs = [key for key in equivalency_map if equivalency_map[key] == tkn and key in initial_mm.asset_list]
        if eqs:
            start_price[tkn] = sum(start_price[eq] for eq in eqs) / len(eqs)

for exchange in initial_stableswaps:
    priced_tokens = [tkn for tkn in exchange.asset_list if start_price[tkn] != 0]
    if priced_tokens == []:
        for tkn in exchange.asset_list:
            start_price[tkn] = 1  # default price for USD
    else:
        for tkn in exchange.asset_list:
            if tkn not in priced_tokens:
                start_price[tkn] = exchange.price(tkn, start_price[priced_tokens[0]]) * start_price[priced_tokens[0]]

st.session_state["time_steps"] = 10
st.session_state.setdefault("add_collateral", {})
st.session_state.setdefault("add_debt", {})
st.session_state["price_change"] = {
    tkn: price_change_defaults[tkn] for tkn in price_change_defaults if price_change_defaults[tkn] != 0
}
st.session_state["debt_totals"] = {
    tkn: sum([cdp.debt[tkn] for cdp in initial_mm.cdps if tkn in cdp.debt]) for tkn in initial_mm.borrowed
}
st.session_state["collateral_totals"] = {
    tkn: sum([cdp.collateral[tkn] for cdp in initial_mm.cdps if tkn in cdp.collateral]) for tkn in initial_mm.asset_list
}
print(st.session_state["debt_totals"])
st.session_state["run_simulation"] = False
st.session_state.setdefault("money_market", initial_mm.copy())

with st.sidebar:
    def sidebar_builder():
        label_col, input_col = st.columns([2, 1], vertical_alignment="center")
        with label_col:
            st.subheader("Time steps:")
        with input_col:
            st.session_state["time_steps"] = st.number_input(
                label="time steps",
                min_value=1,
                max_value=100,
                value=st.session_state["time_steps"],
                label_visibility="collapsed"
            )

        def change_param_form(
                key: str,
                title: str,
                default_value: float,
                default_token: str,
                min_value: float,
                max_value: float,
                on_change: Callable,
                expanded: bool = True,
                as_percentage: bool = False
        ):
            def set_delete_flag(asset, flag):
                st.session_state.__setitem__(flag, asset)
            expanded = st.session_state.get(f"{key}_expander", expanded)
            st.session_state[f"{key}_expander"] = expanded

            delete_flag = f"{key}_pending_delete"
            asset_to_delete = st.session_state.pop(delete_flag, None)
            if asset_to_delete is not None:
                st.session_state[key].pop(asset_to_delete, None)
                on_change()
            with st.expander(title, expanded=expanded):
                with st.form(f"{key}_change_form", border=False):
                    label_column, input_column = st.columns([3, 2], vertical_alignment="center")
                    with label_column:
                        st.selectbox(" ", main_tokens, index=main_tokens.index(default_token), key=f"{key}_asset",
                                     label_visibility="collapsed")
                    with input_column:
                        st.number_input(
                            label=key,
                            min_value=min_value,
                            max_value=max_value,
                            value=default_value,
                            key=f"{key}_amt",
                            label_visibility="collapsed"
                        )
                    with st.columns([1, 3, 1])[1]:
                        add_p = form_submit_button("Add", use_container_width=True)
                    if add_p:
                        if st.session_state[f"{key}_asset"] not in st.session_state[key]:
                            st.session_state[key][st.session_state[f"{key}_asset"]] = 0
                        st.session_state[key][st.session_state[f"{key}_asset"]] += float(
                            st.session_state.get(f"{key}_amt", 0) or 0)
                        on_change()

                for asset, val in sorted(st.session_state[key].items()):
                    name_col, amt_col, del_col = st.columns([5, 3, 2])
                    with name_col:
                        st.write(asset)
                    with amt_col:
                        if as_percentage:
                            st.write(f"{"+" if val > 0 else ""}{val:,.0f}%")
                        else:
                            st.write(f"{val:,.0f}")
                    with del_col:
                        st.button(
                            "✕",
                            key=f"del_{key}_{asset}",
                            on_click=lambda flag=delete_flag: set_delete_flag(asset, flag),
                        )

        def show_asset_config(asset: MoneyMarketAsset, title: str=None):
            with st.container():
                name = asset.name
                title = title or name
                st.markdown(
                    f"""
                    <div style="
                        text-align: center;
                        border-top: 1px dotted #ccc;
                    ">
                        <h3 style="color: #886b6b; margin: 0;">{title}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("Price (USD):")
                with input_col:
                    asset.price = st.number_input(
                        label=f"price ({name})",
                        value=asset.price,
                        min_value=0.0000001,
                        max_value=1_000_000.0,
                        format=f"%.{min(6, 6-math.floor(math.log10(asset.price)))}f",
                        label_visibility="collapsed"
                    )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("Liquidation threshold:")
                with input_col:
                    asset.liquidation_threshold = st.number_input(
                        label=f"liquidation threshold ({name})",
                        min_value=0.5,
                        max_value=0.99,
                        value=asset.liquidation_threshold,
                        label_visibility="collapsed"
                    )
                label_col, input_col = st.columns([3, 2], vertical_alignment="center")
                with label_col:
                    st.write("Liquidation bonus:")
                with input_col:
                    asset.liquidation_bonus = st.number_input(
                        label=f"liquidation bonus ({name})",
                        min_value=0.01,
                        max_value=0.2,
                        value=asset.liquidation_bonus,
                        label_visibility="collapsed"
                    )

        def money_market_config_section():
            title = "Money Market Parameters"
            new_assets = st.session_state.get("new_assets", set())
            mm_sim = st.session_state["money_market"]
            expander_key = "mm_config_expander"
            is_expanded = st.session_state.get(expander_key, False)

            with st.expander(title, expanded=is_expanded):
                for name, asset in mm_sim.assets.items():
                    is_new = name in new_assets
                    if not is_new:
                        show_asset_config(asset)

        def update_money_market_assets():
            new_assets = {}
            for tkn in set(st.session_state.add_collateral.keys()).union(st.session_state.add_debt.keys()):
                if tkn not in mm.asset_list:
                    new_assets[tkn] = MoneyMarketAsset(
                        name=tkn,
                        price=start_price[tkn],
                        liquidation_threshold=0.7,
                        liquidation_bonus=0.05,
                    )
            if new_assets != {}:
                mm.asset_list += list(new_assets.keys())
                mm.assets.update(new_assets)
                print(f"Added new assets to money market: {list(new_assets.keys())}")
                st.session_state["new_assets"] = st.session_state.get("new_assets", set()) | set(new_assets.keys())
                st.rerun()

        def sum_debt():
            update_money_market_assets()
            st.session_state.debt_totals = {
                tkn: sum([cdp.debt[tkn] for cdp in mm.cdps if tkn in cdp.debt])
                + st.session_state.get("add_debt", {}).get(tkn, 0)
                for tkn in list(set(mm.borrowed.keys()).union(set(st.session_state.get("add_debt", {}).keys())))
            }
            print(st.session_state.debt_totals)

        def sum_collateral():
            update_money_market_assets()
            st.session_state.collateral_totals = {
                tkn: sum([cdp.collateral[tkn] for cdp in mm.cdps if tkn in cdp.collateral])
                + st.session_state.get("add_collateral", {}).get(tkn, 0)
                for tkn in list(set(mm.asset_list).union(set(st.session_state.get("add_collateral", {}).keys())))
            }
            print("new collateral:", st.session_state.get("add_collateral", {}))
            print(st.session_state.collateral_totals)

        change_param_form(
            key="price_change",
            as_percentage=True,
            default_token="HDX",
            default_value=-20,
            min_value=-99,
            max_value=1000,
            title=f"Price change over {st.session_state['time_steps']} steps",
            on_change=lambda: None,
            expanded=True
        )

        # prompt to enter params for newly added assets
        for asset in st.session_state.get("new_assets", []):
            with st.container():
                show_asset_config(mm.assets[asset], title=f"New asset: {asset}")
                with st.columns([1, 3, 1])[1]:
                    remove_p = st.button("Ok", use_container_width=True)
                if remove_p:
                    st.session_state.money_market.prices[asset] = mm.assets[asset].price
                    st.session_state.new_assets.remove(asset)
                    st.rerun()

        change_param_form(
            key="add_collateral",
            title="Add collateral",
            default_token="HDX",
            default_value=80_000_000,
            min_value=0,
            max_value=1_000_000_000,
            on_change=sum_collateral,
            expanded=False
        )
        change_param_form(
            key="add_debt",
            title="Add debt",
            default_token="USD",
            default_value=1_000_000,
            min_value=0,
            max_value=1_000_000_000,
            on_change=sum_debt,
            expanded=False
        )
        money_market_config_section()

    sidebar_builder()

def run_app():
    price_factor = {
        tkn: st.session_state["price_change"].get(tkn, 0) for tkn in main_tokens
    }
    # update price change defaults for equivalent tokens
    price_factor.update({
        tkn: price_factor[equivalency_map[tkn]] for tkn in equivalency_map.keys()
        if equivalency_map[tkn] in main_tokens
    })
    # trade_agent = Agent(enforce_holdings=False)
    def update_prices(state: GlobalState):
        for price_tkn in price_paths:
            relevant_pools = [pool for pool in state.pools.values() if price_tkn in pool.asset_list]
            for pool in relevant_pools:
                if isinstance(pool, FixedPriceExchange):
                    pool.prices[price_tkn] = price_paths[price_tkn][state.time_step - 1]
                elif isinstance(pool, MoneyMarket):
                    pool.assets[price_tkn].price = price_paths[price_tkn][state.time_step - 1]
                    pool.prices[price_tkn] = price_paths[price_tkn][state.time_step - 1]
                # elif isinstance(pool, OmnipoolState):
                #     # execute trades to move price towards target
                #     target_price = price_paths[price_tkn][state.time_step - 1] * pool.lrna_price('USD')
                #     pool.trade_to_price(trade_agent, tkn=price_tkn, target_price=target_price)

            state.external_market[price_tkn] = price_paths[price_tkn][state.time_step - 1]

    time_steps = st.session_state["time_steps"]
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

    lrna_price_usd = initial_omnipool.lrna_price('USD')
    omnipool_swap_quantities = {
        tkn: [
            initial_omnipool.calculate_trade_to_price(tkn=tkn, target_price=price_paths[tkn][i] * lrna_price_usd)
            for i in range(time_steps)
        ] for tkn in all_price_change_tokens if tkn in initial_omnipool.asset_list and price_paths[tkn][-1] != initial_omnipool.usd_price(tkn)
    }
    swap_schedule = []
    for i in range(time_steps):
        swap_schedule.append([])
        for tkn in omnipool_swap_quantities:
            net_swap = omnipool_swap_quantities[tkn][i] - (omnipool_swap_quantities[tkn][i - 1] if i > 0 else 0)
            swap_schedule[i].append({
                'tkn_sell': tkn if net_swap > 0 else 'LRNA',
                'tkn_buy': 'LRNA' if net_swap > 0 else tkn,
                'sell_quantity': net_swap if net_swap > 0 else None,
                'buy_quantity': -net_swap if net_swap < 0 else None,
            })
    #
    # config_list = [
    #     {'exchanges': {'router': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
    #     {'exchanges': {'omnipool': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
    # ]
    omnipool = copy.deepcopy(initial_omnipool)
    stableswaps = [exchange.copy() for exchange in initial_stableswaps]
    router = initial_router.copy()
    # reset money market CDPS to initial state
    mm.cdps = [cdp.copy() for cdp in initial_mm.cdps]
    # add collateral and debt to money market
    router.exchanges = {pool.unique_id: pool for pool in [omnipool, mm, *stableswaps]}

    initial_state = GlobalState(
        pools=[router, omnipool, mm, *stableswaps],
        agents={
            'liquidator': Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('router')),
            'trader': Agent(
                enforce_holdings=False,
                trade_strategy=schedule_swaps(
                    swaps=swap_schedule,
                    pool_id='omnipool'
                )
            ),
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
                if tkn in event.pools['money_market'].asset_list
                else event.pools['router'].price(tkn, 'USD')
                for event in events
            ]
            if max(tkn_price_path) == 0:
                continue
            if abs(min(tkn_price_path) / max(tkn_price_path) - 1) < 0.01:
                continue
            # if abs(1 - tkn_price_path[-1] / final_price[tkn]) > 0.01:
            #     st.warning(f"{tkn} price did not reach expected final price of {final_price[tkn]} USD")
            ax.plot(tkn_price_path, label=f"{tkn} price")
            d = max(0, 2 - int(math.floor(math.log10(abs(tkn_price_path[0])))))
            p = tkn_price_path[0]
            ax.annotate(
                f"${(int(p) if d==0 else f'{p:.2f}' if d<=2 else round(p, d))}",
                xy=(0, tkn_price_path[0]),
                xytext=(0, tkn_price_path[0] * (1.05 if tkn_price_path[0] < tkn_price_path[-1] else 0.95))
            )
            ax.scatter(marker='o', s=20, x=0, y=tkn_price_path[0], color='#009')
            p = tkn_price_path[-1]
            ax.annotate(
                f"${(int(p) if d==0 else f'{p:.2f}' if d<=2 else round(p, d))}",
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

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Run simulation", use_container_width=True):
        st.session_state.run_simulation = True


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

plot_health_factor_distribution(initial_mm)


def plot_cdp_value_distribution():
    with st.expander(f"Initial debt and collateral token amounts"):
        (token_name, collateral_amt, debt_amt) = st.columns(3)
        token_name.markdown("**Token**")
        collateral_amt.markdown("**Total Collateral**")
        debt_amt.markdown("**Total Debt**")
        asset_list = list(set(st.session_state["debt_totals"].keys()).union(set(st.session_state["collateral_totals"].keys())))
        for tkn in asset_list:
            with token_name:
                one_line_markdown(f"**{tkn}**")
                one_line_markdown("-------------------")
            with debt_amt:
                if tkn in st.session_state.debt_totals:
                    one_line_markdown(
                        f"{st.session_state.debt_totals[tkn]:,.2f} ({st.session_state.debt_totals[tkn] * mm.assets[tkn].price:,.2f} USD)"
                    )
                else:
                    one_line_markdown("0")
                one_line_markdown("-------------------")

            with collateral_amt:
                if tkn in st.session_state.collateral_totals:
                    one_line_markdown(
                        f"{st.session_state.collateral_totals[tkn]:,.2f} ({st.session_state.collateral_totals[tkn] * mm.assets[tkn].price:,.2f} USD)"
                    )
                else:
                    one_line_markdown("0")
                one_line_markdown("-------------------")

plot_cdp_value_distribution()

if st.session_state.run_simulation:
    # plot_health_factor_distribution(mm_sim)
    # plot_cdp_value_distribution(mm_sim)
    run_app()
    st.session_state.run_simulation = False