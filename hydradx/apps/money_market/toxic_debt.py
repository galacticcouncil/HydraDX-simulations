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
from hydradx.model.amm.trade_strategies import liquidate_cdps, TradeStrategy, general_arbitrage
from hydradx.model.processing import get_current_money_market, get_stableswap_data, Pool
from hydradx.model.amm.agents import Agent
from hydradx.model.amm.fixed_price import FixedPriceExchange
from hydradx.model.run import run
from hydradx.model.amm.trade_strategies import constant_swaps
from hydradx.model.indexer_utils import get_current_omnipool_router, get_asset_info

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
    load_router = get_current_omnipool_router()
    load_omnipool = load_router.exchanges['omnipool']

    asset_info = get_asset_info()
    stable_swap_data = get_stableswap_data()
    stableswap_pools = []
    usd_price_lrna = (
        1 / load_omnipool.lrna_price('2-Pool-Stbl') / stable_swap_data[102].shares * 10 ** 18
        * sum([int(v) / 10 ** asset_info[k].decimals for k, v in stable_swap_data[102].reserves.items()])
    )
    load_omnipool.add_token(
        'USD', liquidity = usd_price_lrna, lrna=1
    ).stablecoin = 'USD'
    for pool in stable_swap_data.values():
        pool_name = asset_info[pool.pool_id].name
        if pool.pool_id == 690:
            peg = load_omnipool.lrna_price('vDOT') / load_omnipool.lrna_price('DOT')
            shares = pool.shares / 10 ** 18
            tokens = {
                'DOT': shares * peg / (1 + peg),
                'vDOT': shares / (1 + peg),
            }
        else:
            tokens={
                asset_info[tkn_id].symbol:
                    int(pool.reserves[tkn_id]) / 10 ** asset_info[tkn_id].decimals
                for tkn_id in pool.reserves
            }
            peg = None
        stableswap_pools.append(
            StableSwapPoolState(
                tokens=tokens,
                peg=peg,
                amplification=pool.final_amplification,
                trade_fee=pool.fee,
                unique_id=pool_name,
            )
        )

    mm = get_current_money_market()
    # mm = MoneyMarket(
    #     assets=[
    #         MoneyMarketAsset(
    #             'DOT', load_omnipool.usd_price('DOT'), 0.7, 0.01, 0.01
    #         ),
    #         MoneyMarketAsset(
    #             'vDOT', load_omnipool.usd_price('vDOT'), 0.7, 0.01, 0.01
    #         ),
    #         MoneyMarketAsset(
    #             name='2-Pool-GDOT',
    #             price=(load_omnipool.usd_price('DOT') + load_omnipool.usd_price('vDOT')) / 2,
    #             liquidation_threshold=0.7,
    #             liquidation_bonus=0.01
    #         ),
    #         MoneyMarketAsset(
    #             name='USD',
    #             price=1,
    #             liquidation_threshold=0.8,
    #             liquidation_bonus=0.01
    #         )
    #     ],
    #     unique_id='money_market',
    #     cdps=[
    #         CDP(debt={'USD': 1000}, collateral={'DOT': 500})
    #     ]
    # )

    print("Finished downloading data.")
    return OmnipoolRouter(exchanges=[load_omnipool, mm, *stableswap_pools], unique_id='router'), cache_time


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


def update_prices(state):
    # prices = {
    #     tkn: state.pools['omnipool'].usd_price(tkn)
    #     for tkn in list(set(state.pools['money_market'].prices) & set(omnipool.asset_list))
    # }
    for tkn in prices:
        state.pools['money_market'].prices[tkn] = prices[tkn][state.time_step - 1]
        state.pools['binance'].prices[tkn] = prices[tkn][state.time_step - 1]


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
omnipool = router.exchanges['omnipool']
mm = router.exchanges['money_market']
stableswaps = [exchange for exchange in router.exchanges.values() if isinstance(exchange, StableSwapPoolState)]
st.sidebar.info(f"Data loaded at: {cache_timestamp}")

price_change_defaults = {
    tkn: 0 for tkn in mm.asset_list
}
price_change_defaults.update({
    'DOT': -75,
    'HDX': 50
})
start_price = {
    tkn: omnipool.usd_price(tkn) if tkn in omnipool.asset_list else mm.price(tkn)
    for tkn in mm.asset_list
}

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

# calculate price path from start price to crash price
final_price = {tkn: start_price[tkn] * (1 + price_factor[tkn] / 100) for tkn in start_price}
prices = {
    tkn: [
        start_price[tkn]] + [start_price[tkn] - (start_price[tkn] - final_price[tkn]) * ((i + 1) / time_steps)
        for i in range(time_steps)
    ] for tkn in start_price
}
full_trades = {tkn: [trade_to_price(omnipool, tkn, price) for price in prices[tkn]] for tkn in start_price}
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
        } for tkn in list(set(omnipool.asset_list) & set(mm.asset_list))
    ] for i in range(1, time_steps + 1)
]
trade_sequence.extend([[
    {'tkn_sell': tkn, 'tkn_buy': 'LRNA', 'sell_quantity': 0}
    for tkn in list(set(omnipool.asset_list) & set(mm.asset_list))
]])
time_steps = len(trade_sequence)
omnipool_sim = copy.deepcopy(omnipool)
mm_sim = mm.copy()
binance = FixedPriceExchange(
    tokens={
        tkn: omnipool.usd_price(tkn)
        if tkn in omnipool.asset_list
        else mm.price(tkn)
        for tkn in list(set(omnipool.asset_list + mm.asset_list))
    }, unique_id='binance'
)

config_list = [
    {'exchanges': {'omnipool': ['DOT', 'vDOT'], '2-Pool-GDOT': ['aDOT', 'vDOT']}, 'buffer': 0.001},
]

initial_state = GlobalState(
    pools=[router, mm_sim, omnipool_sim, binance, *stableswaps],
    agents={
        'liquidator': Agent(enforce_holdings=False, trade_strategy=liquidate_cdps('omnipool')),
        'panic seller': Agent(
            enforce_holdings=False,
            trade_strategy=schedule_swaps(trade_sequence)
        ),
        'arbitrageur': Agent(
            enforce_holdings=False,
            trade_strategy=general_arbitrage(
                exchanges=[omnipool, stableswaps[-1]],
                config=config_list
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

for tkn in start_price:
    fig, ax = plt.subplots()
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(f"{tkn} price (USD)")
    ax.plot([
        event.pools['omnipool'].usd_price(tkn)
        if tkn in event.pools['omnipool'].asset_list
        else event.pools['money_market'].price(tkn)
        for event in events
    ])
    st.pyplot(fig)
