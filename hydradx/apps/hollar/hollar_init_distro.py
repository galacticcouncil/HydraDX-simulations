from matplotlib import pyplot as plt
import multiprocessing as mp
import functools as ft
import sys, os
import streamlit as st

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from hydradx.model.amm.stableswap_amm import StableSwapPoolState
from hydradx.model.amm.agents import Agent
from hydradx.model.hollar import StabilityModule
from hydradx.apps.display_utils import display_ss_multiple

# hardcoded values
PEGS = {'aUSDT': 1, 'aUSDC': 1, 'sUSDS': 1.05, 'sUSDE': 1.16}
DEFAULT_HOLLAR_LIQ = 2_000_000
DEFAULT_AMP = 50
SWAP_FEE = 0.0002
HSM_LIQUIDITY = {'aUSDT': 500_000, 'aUSDC': 500_000, 'sUSDS': 500_000 / PEGS['sUSDS'], 'sUSDE': 500_000 / PEGS['sUSDE']}
DEFAULT_BUYBACK_SPEED = 0.0002
DEFAULT_SIMULATE_BLOCKS = 40_000
DEFAULT_HOLLAR_AMOUNTS = [1000000, 1000000, 1000000, 1000000, 1000000]
DEFAULT_DUMP_BLOCKS = [100, 500, 1000, 3000, 5000]
TKN_CHART = 'aUSDT'
PRICE_PLOT_N = 100  # how frequently are we calculating spot price to graph it

def collect_inputs() -> dict:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                min-width: 400px;
                max-width: 800px;
            }
            .stNumberInput button {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    inputs = {}
    inputs['init_hollar_liq'] = st.sidebar.number_input(
        "Initial Hollar liquidity",
        min_value=100_000, value=DEFAULT_HOLLAR_LIQ, step=1, key="init_hollar_liq"
    )

    inputs['amp'] = st.sidebar.number_input(
        "Amplification for stableswap pools",
        min_value=5, value=DEFAULT_AMP, step=1, key="amp"
    )

    col1, col2 = st.columns(2)
    with col1:
        inputs['buyback_speed'] = st.number_input(
            "Buyback speed",
            min_value=0.000001, value=DEFAULT_BUYBACK_SPEED, step=0.000001, key="buyback_speed", format="%.6f",
            help=("This number will be multiplied by the imbalance in the associated stableswap pool to determine how"
                  " much Hollar can be bought back in one block. If a pool has 1,000,000 Hollar and 100_000 aUSDT, the"
                  " imbalance is 450_000. That number would be multiplied by the buyback speed to determine how much Hollar"
                  " can be bought back in one block.")
        )

    with col2:
        inputs['num_blocks'] = st.number_input(
            "Number of blocks to simulate",
            min_value=1000, value=DEFAULT_SIMULATE_BLOCKS, step=1, key="num_blocks",
            help="Simulation will end early if HSM has brought Hollar price back to peg"
        )

    st.subheader("Scenarios")

    if "num_pairs" not in st.session_state:
        st.session_state.num_pairs = 5  # default starting value

    col1, col2 = st.columns(2)
    if st.session_state.num_pairs < 10:
        with col1:
            scenario_added = st.button("Add scenario")
            if scenario_added:
                st.session_state.num_pairs += 1
    if st.session_state.num_pairs > 1:
        with col2:
            scenario_removed = st.button("Remove last scenario")
            if scenario_removed:
                st.session_state.num_pairs -= 1

    n = st.session_state.num_pairs
    hollar_amounts = DEFAULT_HOLLAR_AMOUNTS[:n] + [DEFAULT_HOLLAR_AMOUNTS[-1]] * max(0, n - len(DEFAULT_HOLLAR_AMOUNTS))
    hollar_dump_blocks = DEFAULT_DUMP_BLOCKS[:n] + [DEFAULT_DUMP_BLOCKS[-1]] * max(0, n - len(DEFAULT_DUMP_BLOCKS))
    inputs['hollar_amounts_inputs'] = []
    inputs['hollar_dump_blocks_inputs'] = []

    with col1:
        st.text("Hollar Dumped")
    with col2:
        st.text("Duration of Hollar dump (in blocks)")

    for i in range(st.session_state.num_pairs):
        with col1:
            inputs['hollar_amounts_inputs'].append(st.number_input(
                f"sell_amt_{i}", min_value=0, value=hollar_amounts[i], step=1, key=f"sell_amt_{i}",
                label_visibility='collapsed'
            ))
        with col2:
            inputs['hollar_dump_blocks_inputs'].append(st.number_input(
                f"sell_blocks_{i}", min_value=0, value=hollar_dump_blocks[i], step=1, key=f"sell_blocks_{i}",
                label_visibility='collapsed'
            ))
    return inputs

# caching function for different scenarios
@st.cache_data
def simulate_scenario(
        init_hollar_liq: float,
        amp: int,
        buyback_speed: float,
        num_blocks: int,
        sell_amt: float,
        num_blocks_dump: int
) -> dict:
    init_stablepools = {}
    for tkn in ['aUSDT', 'aUSDC', 'sUSDS', 'sUSDE']:
        stable_tokens = {tkn: init_hollar_liq / 4, 'HOLLAR': init_hollar_liq / 4}
        init_stablepools[tkn] = StableSwapPoolState(
            stable_tokens, amp, trade_fee=SWAP_FEE, unique_id=tkn, peg=PEGS[tkn]
        )

    ss_liquidity = [ss.liquidity for ss in init_stablepools.values()]
    init_pools_list = [init_stablepools[tkn] for tkn in HSM_LIQUIDITY]

    stablepools = {TKN_CHART: init_stablepools[TKN_CHART].copy()}
    pools_list = [stablepools[TKN_CHART]]
    reduced_liquidity = {TKN_CHART: HSM_LIQUIDITY[TKN_CHART]}
    hsm = StabilityModule(reduced_liquidity, buyback_speed, pools_list, max_buy_price_coef=0.999)
    arb_agent = Agent(enforce_holdings=False)
    spot_prices = [hsm.pools[TKN_CHART].price('HOLLAR', TKN_CHART)]
    init_hsm_value = sum([PEGS[tkn] * HSM_LIQUIDITY[tkn] for tkn in PEGS])
    hsm_values = [init_hsm_value]
    hollar_sell_amts = []
    for j in range(num_blocks):
        if (len(hsm_values) > num_blocks_dump and (hsm_values[-1] == hsm_values[-2]
                                                   or spot_prices[-1] >= hsm.max_buy_price_coef[
                                                       TKN_CHART])):  # note this only works because aUSDT peg is 1
            len_extend = min(num_blocks - j, 1000)
            hsm_values.extend([hsm_values[-1]] * len_extend)
            spot_prices.extend([spot_prices[-1]] * (len_extend // PRICE_PLOT_N))
            hollar_sell_amts.extend([0] * len_extend)
            break
        before_hsm_liq = hsm.liquidity[TKN_CHART]
        for tkn, ss in hsm.pools.items():
            max_buy_amt = hsm._get_max_buy_amount(tkn)  # note this ignores self.max_buy_price_coef
            if tkn == TKN_CHART:
                hollar_sold = max_buy_amt  # track max_buy_amt for aUSDT
            if j < num_blocks_dump:  # add in Hollar dumping to net swap
                hollar_buy_amt = max_buy_amt - sell_amt / num_blocks_dump / 4
            else:
                hollar_buy_amt = max_buy_amt

            arb_agent.add(hsm.native_stable, max_buy_amt)  # flash mint Hollar for arb
            hsm.swap(arb_agent, tkn_buy=tkn, tkn_sell=hsm.native_stable, sell_quantity=max_buy_amt)
            if hollar_buy_amt > 0:
                ss.swap(arb_agent, tkn_buy=hsm.native_stable, tkn_sell=tkn, buy_quantity=hollar_buy_amt)
            elif hollar_buy_amt < 0:
                ss.swap(arb_agent, tkn_buy=tkn, tkn_sell=hsm.native_stable, sell_quantity=-hollar_buy_amt)
            arb_agent.remove(hsm.native_stable, max_buy_amt)  # burn Hollar that was minted

            ss.update()
            hsm.update()
        after_hsm_liq = hsm.liquidity[TKN_CHART]
        hsm_delta = after_hsm_liq - before_hsm_liq
        hsm_loss_total = hsm_delta * len(init_stablepools)
        if (j + 1) % PRICE_PLOT_N == 0:
            spot_prices.append(hsm.pools[TKN_CHART].price('HOLLAR', TKN_CHART))
        hsm_values.append(hsm_values[-1] + hsm_loss_total)
        hollar_sell_amts.append(hollar_sold)

    results = {'ss_liquidity': ss_liquidity}
    results['spot_prices'] = spot_prices
    results['hsm_values'] = hsm_values
    results['hollar_sold'] = hollar_sell_amts
    return results

def wrapper(args, init_hollar_liq, amp, buyback_speed, num_blocks):
    sell_amt, num_blocks_dump = args
    return simulate_scenario(
        init_hollar_liq,
        amp,
        buyback_speed,
        num_blocks,
        sell_amt,
        num_blocks_dump
    )

def get_results(inputs: dict) -> tuple:
    # prepare data for simulation function
    init_hollar_liq = inputs['init_hollar_liq']
    amp = inputs['amp']
    buyback_speed = inputs['buyback_speed']
    num_blocks = inputs['num_blocks']
    hollar_amounts_inputs = inputs['hollar_amounts_inputs']
    hollar_dump_blocks_inputs = inputs['hollar_dump_blocks_inputs']
    hollar_amounts = []
    hollar_dump_blocks = []
    for i in range(len(hollar_amounts_inputs)):
        if hollar_amounts_inputs[i] > 0 and hollar_dump_blocks_inputs[i] > 0:
            hollar_amounts.append(hollar_amounts_inputs[i])
            hollar_dump_blocks.append(hollar_dump_blocks_inputs[i])

    hsm_vals_dict = {}
    spot_prices_dict = {}
    hollar_sold_dict = {}
    ss_liquidity_dict = {}
    scenario_data = list(zip(hollar_amounts, hollar_dump_blocks))

    with mp.Pool() as pool:
        results = pool.map(
            ft.partial(
                wrapper,
                init_hollar_liq=init_hollar_liq,
                amp=amp,
                buyback_speed=buyback_speed,
                num_blocks=num_blocks
            ),
            scenario_data
        )
    for i in range(len(scenario_data)):
        ss_liquidity_dict[scenario_data[i]] = results[i]['ss_liquidity']
        spot_prices_dict[scenario_data[i]] = results[i]['spot_prices']
        hsm_vals_dict[scenario_data[i]] = results[i]['hsm_values']
        hollar_sold_dict[scenario_data[i]] = results[i]['hollar_sold']
    return ss_liquidity_dict, spot_prices_dict, hsm_vals_dict, hollar_sold_dict


def graph_results(ss_liquidity_dict, spot_prices_dict, hsm_vals_dict, hollar_sold_dict):
    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), spot_prices in spot_prices_dict.items():  # interpolate prices
        interp_spot_prices = []
        for i in range(len(spot_prices) - 1):
            p = spot_prices[i]
            next_p = spot_prices[i+1]
            interped_prices = [p + (next_p - p) * j / PRICE_PLOT_N for j in range(PRICE_PLOT_N)]
            interp_spot_prices.extend(interped_prices)

        ax.plot(interp_spot_prices, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
    ax.set_title("Spot price of Hollar (in aUSDT)")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), hsm_values in hsm_vals_dict.items():
        ax.plot(hsm_values, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
    ax.set_title("Value remaining in Stability Module")
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    for (sell_amt, num_blocks_dump), hollar_sell_amts in hollar_sold_dict.items():
        ax.plot(hollar_sell_amts, label=f"{sell_amt} Hollar, {num_blocks_dump} blocks")
    ax.set_title("Amount of Hollar sold per block, for aUSDT")
    ax.legend()
    st.pyplot(fig)

    st.sidebar.subheader("Liquidity distribution")
    fig = display_ss_multiple(list(ss_liquidity_dict.values())[0])
    st.sidebar.pyplot(fig)

def run_script():
    inputs = collect_inputs()
    ss_liquidity_dict, spot_prices_dict, hsm_vals_dict, hollar_sold_dict = get_results(inputs)
    graph_results(ss_liquidity_dict, spot_prices_dict, hsm_vals_dict, hollar_sold_dict)

if __name__ == "__main__":
    run_script()
