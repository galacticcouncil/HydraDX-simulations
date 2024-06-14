from hydradx.model.amm.omnipool_amm import OmnipoolState
from hydradx.model.amm.agents import Agent
import copy
from mpmath import mp, mpf



def validate_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        lrna_swaps: dict,  # key: token, value: delta_change of LRNA w.r.t. Omnipool
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    return validate_and_execute_solution(
        omnipool=omnipool.copy(),
        intents=copy.deepcopy(intents),
        amt_processed=amt_processed,
        lrna_swaps=lrna_swaps,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        tolerance=tolerance
    )


def validate_and_execute_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        lrna_swaps: dict,  # key: token, value: delta_change of LRNA w.r.t. Omnipool
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        tolerance: float = 0.01
):
    assert buy_prices.keys() == sell_prices.keys(), "buy_prices and sell_prices are not provided for same tokens"
    assert buy_prices.keys() == lrna_swaps.keys(), "buy_prices and lrna_swaps are not provided for same tokens"

    validate_intents(intents, amt_processed, lrna_swaps)
    pool_agent, deltas = execute_solution(omnipool, intents, amt_processed, lrna_swaps, buy_prices, sell_prices)
    if not validate_remainder(pool_agent):
        raise Exception("agent has negative holdings")
    if not validate_prices(omnipool, lrna_swaps, buy_prices, sell_prices, deltas, tolerance):
        raise Exception("prices are not valid")

    return True


def validate_intents(intents: list, amt_processed: list, lrna_swaps: dict):
    for i in range(len(intents)):
        intent = intents[i]
        amt = amt_processed[i]
        if 'sell_quantity' in intent:
            assert "buy_limit" in intent, "buy_limit not in sell intent"
            assert "buy_quantity" not in intent, "intent has both buy_quantity and sell_quantity"
            assert "sell_limit" not in intent, "intent has both buy_quantity and sell_limit"
            assert 0 <= amt <= intent['sell_quantity'], "amt processed is not in range"
        elif 'buy_quantity' in intent:
            assert "sell_limit" in intent, "sell_limit not in buy intent"
            assert "buy_limit" not in intent, "intent has both sell_quantity and buy_limit"
            assert 0 <= amt <= intent['buy_quantity'], "amt processed is not in range"
        else:
            raise Exception("intent does not have sell_quantity or buy_quantity")

    return True


def execute_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        amt_processed: list,  # list of amt processed for each intent
        lrna_swaps: dict,  # key: token, value: delta_change of LRNA w.r.t. Omnipool
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict  # key: token, value: price
):
    # pool assets coming in/out

    # buy intent is dict with keys: agent, tkn_buy, tkn_sell, buy_quantity, sell_limit, partial
    # sell intent is dict with keys: agent, tkn_sell, tkn_buy, sell_quantity, buy_limit, partial
    agent_deltas_out = {tkn: 0 for tkn in buy_prices}
    agent_deltas_in = {tkn: 0 for tkn in buy_prices}
    for i in range(len(intents)):
        intent = intents[i]
        amt = amt_processed[i]
        if amt > 0:
            if 'sell_quantity' in intent:
                if intent['agent'].is_holding(intent['tkn_sell'], amt):
                    agent_deltas_in[intent['tkn_sell']] += amt
                    intent['agent'].holdings[intent['tkn_sell']] -= amt
                else:
                    raise Exception("agent does not have enough holdings")
            elif 'buy_quantity' in intent:
                agent_deltas_out[intent['tkn_buy']] += amt
                if not intent['agent'].is_holding(intent['tkn_buy']):
                    intent['agent'].holdings[intent['tkn_buy']] = 0
                intent['agent'].holdings[intent['tkn_buy']] += amt

    pool_agent = Agent(holdings={tkn: agent_deltas_in[tkn] - agent_deltas_out[tkn] for tkn in agent_deltas_out})
    pool_agent.holdings["LRNA"] = 0
    # agent_deltas_out = {tkn: -amt if amt < 0 else 0 for tkn, amt in intent_deltas.items()}
    # agent_deltas_in = {tkn: amt if amt > 0 else 0 for tkn, amt in intent_deltas.items()}

    # trade assets for LRNA
    init_liquidity = {tkn: omnipool.liquidity[tkn] for tkn in buy_prices}
    for tkn, delta_lrna in lrna_swaps.items():
        dummy_agent = Agent(holdings={tkn: mpf(omnipool.liquidity[tkn])})  # TODO: better solution
        omnipool.lrna_swap(dummy_agent, delta_qa=-delta_lrna, tkn=tkn)
        pool_agent.holdings[tkn] += dummy_agent.holdings[tkn] - dummy_agent.initial_holdings[tkn]
        pool_agent.holdings["LRNA"] -= delta_lrna
    omnipool_deltas = {tkn: omnipool.liquidity[tkn] - init_liquidity[tkn] for tkn in buy_prices}

    # distribute to users
    for i in range(len(intents)):
        intent = intents[i]
        amt = amt_processed[i]
        if amt > 0:
            if 'sell_quantity' in intent:  # distribute from price
                buy_amt = amt * sell_prices[intent['tkn_sell']] / buy_prices[intent['tkn_buy']]
                limit_price = intent['sell_quantity'] / intent['buy_limit']
                assert amt / buy_amt <= limit_price, "buy_limit not satisfied"
                if not intent['agent'].is_holding(intent['tkn_buy']):
                    intent['agent'].holdings[intent['tkn_buy']] = 0
                intent['agent'].holdings[intent['tkn_buy']] += buy_amt
                pool_agent.holdings[intent['tkn_buy']] -= buy_amt
                agent_deltas_out[intent['tkn_buy']] += buy_amt
            elif 'buy_quantity' in intent:
                sell_amt = amt * buy_prices[intent['tkn_buy']] / sell_prices[intent['tkn_sell']]
                limit_price = intent['buy_quantity'] / intent['sell_limit']
                assert amt / sell_amt >= limit_price, "sell_limit not satisfied"
                if intent['agent'].is_holding(intent['tkn_sell'], amt):
                    intent['agent'].holdings[intent['tkn_sell']] -= sell_amt
                    pool_agent.holdings[intent['tkn_sell']] += sell_amt
                    agent_deltas_in[intent['tkn_sell']] += sell_amt
                else:
                    raise Exception("agent does not have enough holdings")

    deltas = {
        'agent_in': agent_deltas_in,
        'agent_out': agent_deltas_out,
        'omnipool': omnipool_deltas
    }

    return pool_agent, deltas


def validate_remainder(pool_agent: Agent):
    for tkn, amt in pool_agent.holdings.items():
        if amt < 0:
            # return False
            raise
    return True


def validate_prices(
        omnipool: OmnipoolState,
        lrna_swaps: dict,  # key: token, value: delta_change of LRNA w.r.t. Omnipool
        buy_prices: dict,  # key: token, value: price
        sell_prices: dict,  # key: token, value: price
        deltas: dict,
        tolerance: float = 0.01
):
    for tkn, amt in lrna_swaps.items():  # calculate net of Omnipool leg and other leg in same direction
        if amt > 0:  # LRNA being sold to Omnipool for TKN
            agent_sell_amt = deltas['agent_in'][tkn]
            omnipool_sell_amt = -deltas['omnipool'][tkn]
            agent_lrna_amt = deltas['agent_in'][tkn] * sell_prices[tkn]
            omnipool_lrna_amt = lrna_swaps[tkn]
            net_price = (agent_lrna_amt + omnipool_lrna_amt) / (agent_sell_amt + omnipool_sell_amt)
            assert abs(buy_prices[tkn] - net_price)/net_price < tolerance, "price not valid for " + tkn
        elif amt < 0:  # LRNA being bought from Omnipool for TKN
            agent_buy_amt = deltas['agent_out'][tkn]
            omnipool_buy_amt = deltas['omnipool'][tkn]
            agent_lrna_amt = deltas['agent_out'][tkn] * buy_prices[tkn]
            omnipool_lrna_amt = lrna_swaps[tkn]
            net_price = (agent_lrna_amt - omnipool_lrna_amt) / (agent_buy_amt + omnipool_buy_amt)
            assert abs(sell_prices[tkn] - net_price)/net_price < tolerance, "price not valid for " + tkn
        else:
            spot_price = omnipool.price(omnipool, tkn, "LRNA")
            assert abs(buy_prices[tkn] - spot_price) / spot_price < tolerance, "price not valid for " + tkn
            assert abs(sell_prices[tkn] - spot_price) / spot_price < tolerance, "price not valid for " + tkn

    return True
