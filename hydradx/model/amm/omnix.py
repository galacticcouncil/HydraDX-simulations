from hydradx.model.amm.omnipool_amm import OmnipoolState, simulate_swap
from hydradx.model.amm.agents import Agent
import copy
from mpmath import mp, mpf
import math

from hydradx.model.amm.stableswap_amm import simulate_add_liquidity, simulate_withdraw_asset, simulate_remove_liquidity


def validate_solution(
        omnipool: OmnipoolState,
        intents: list,  # swap desired to be processed
        intent_deltas: list,  # list of deltas for each intent
):
    return validate_and_execute_solution(
        omnipool=omnipool.copy(),
        intents=copy.deepcopy(intents),
        intent_deltas=intent_deltas
    )


def calculate_transfers(
        intents: list,  # swap desired to be processed
        intent_deltas: list  # list of deltas for each intent
) -> (list, dict):
    transfers = []
    deltas = {}
    for i in range(len(intents)):
        intent = intents[i]
        sell_amt = -intent_deltas[i][0]
        buy_amt = intent_deltas[i][1]
        transfer = {
            'agent': intent['agent'],
            'buy_quantity': buy_amt,
            'sell_quantity': sell_amt,
            'tkn_buy': intent['tkn_buy'],
            'tkn_sell': intent['tkn_sell']
        }
        transfers.append(transfer)
        if intent['tkn_buy'] not in deltas:
            deltas[intent['tkn_buy']] = {"in": 0, "out": 0}
        deltas[intent['tkn_buy']]["out"] += buy_amt
        if intent['tkn_sell'] not in deltas:
            deltas[intent['tkn_sell']] = {"in": 0, "out": 0}
        deltas[intent['tkn_sell']]["in"] += sell_amt

    return transfers, deltas


def validate_and_execute_solution(
        omnipool: OmnipoolState,
        stableswap_list: list,
        intents: list,  # swap desired to be processed
        intent_deltas: list,  # list of deltas for each intent
        omnipool_deltas: dict,
        stableswap_deltas: list,
        tkn_profit: str = None
):

    validate_intents(intents, intent_deltas)
    transfers, deltas = calculate_transfers(intents, intent_deltas)
    validate_transfer_amounts(transfers)
    pool_agent, fee_agent, lrna_deltas = execute_solution(omnipool, stableswap_list, transfers, deltas, omnipool_deltas, stableswap_deltas, exit_to=tkn_profit)
    if not validate_remainder(pool_agent):
        raise Exception("agent has negative holdings")

    update_intents(intents, transfers)
    if tkn_profit is not None:
        tkn_list = [tkn for tkn in pool_agent.holdings if tkn != tkn_profit and pool_agent.holdings[tkn] > 0]
        for tkn in tkn_list:
            omnipool.swap(pool_agent, tkn_profit, tkn, sell_quantity=pool_agent.holdings[tkn])
        return True, (pool_agent.holdings[tkn_profit] if tkn_profit in pool_agent.holdings else 0)
    else:
        return True, None


def validate_intents(intents: list, intent_deltas: list):
    for i in range(len(intents)):
        intent = intents[i]
        sell_amt = -intent_deltas[i][0]
        buy_amt = intent_deltas[i][1]
        if not 0 <= sell_amt <= intent['sell_quantity']:
            raise Exception("amt processed is not in range")
        if intent['partial'] == False and sell_amt not in [0, intent['sell_quantity']]:
            raise Exception("intent should not be partially executed")
        if intent['partial'] == False and 0 < buy_amt < intent['buy_quantity']:
            raise Exception("intent should not be partially executed")
        tolerance = 1e-6  # temporarily allowing some tolerance, only for testing purposes
        if sell_amt > 0 and buy_amt / sell_amt < (1 - tolerance) * intent['buy_quantity'] / intent['sell_quantity']:
            raise Exception("price is not within tolerance")

    return True


def validate_transfer_amounts(transfers: list):
    for transfer in transfers:
        if not transfer['agent'].is_holding(transfer['tkn_sell'], transfer['sell_quantity']):
            raise Exception(f"agent does not have enough holdings in token {transfer['tkn_sell']}")

    return True


def mint_to_quantity(pool_agent: Agent, tkn: str, quantity: float, minted: dict):
    if tkn not in pool_agent.holdings:
        pool_agent.holdings[tkn] = 0
    init_amt = pool_agent.holdings[tkn]
    pool_agent.holdings[tkn] = max(init_amt, quantity)
    mint_amt = pool_agent.holdings[tkn] - init_amt
    if tkn not in minted:  # convenient to guarantee tkn is in dict even if value is 0
        minted[tkn] = 0
    minted[tkn] += mint_amt


def burn_excess(pool_agent: Agent, tkn: str, minted: dict):
    burn_amt = min(pool_agent.holdings[tkn], minted[tkn])
    pool_agent.holdings[tkn] -= burn_amt
    minted[tkn] -= burn_amt


def execute_solution(
        omnipool: OmnipoolState,
        stableswap_list: list,
        transfers: list,
        deltas: dict,  # note that net_deltas can be easily reconstructed from transfers
        omnipool_deltas: dict,
        stableswap_deltas: list,
        fee_match: float = 0.0,
        exit_to: str = None
):
    pool_agent = Agent()
    fee_agent = Agent()

    # amm deltas are *feeless*. We need to recalculate the buy amounts to account for asset fees being left behind
    for tkn in omnipool_deltas:
        if omnipool_deltas[tkn] < 0:
            omnipool_deltas[tkn] *= 1 - omnipool.last_fee[tkn]

    for i, ss in enumerate(stableswap_list):
        for j, tkn in enumerate(ss.asset_list):
            if stableswap_deltas[i][j+1] < 0:
                stableswap_deltas[i][j+1] *= 1 - ss.trade_fee

    # we first execute all AMM trades, flash minting assets to pool_agent to enable sells
    minted = {}  # tracking minted assets so that we burn them later

    for i, ss in enumerate(stableswap_list):
        pool_deltas = stableswap_deltas[i]
        if stableswap_deltas[i][0] > 0:  # separate if statement prevents adding & withdrawing due to rounding issues
            while stableswap_deltas[i][0] > 0:  # need to add liquidity
                # construct list of tokens to add liquidity in
                if max(stableswap_deltas[i][1:]) <= 0:
                    break  # can't add any more liquidity
                tkns = []
                highest_pct = 0
                second_highest_pct = 0
                tol = 1e-9
                for j, tkn in enumerate(ss.asset_list):  # calculate highest and second highest pcts
                    if pool_deltas[j+1] > 0:
                        pct = pool_deltas[j+1] / ss.liquidity[tkn]
                        rel_diff = (pct - highest_pct)/highest_pct if highest_pct > 0 else float("inf")
                        if rel_diff > tol:
                            second_highest_pct = highest_pct
                            highest_pct = pct
                            tkns = [tkn]
                        elif rel_diff > -tol:
                            tkns.append(tkn)
                pct_diff = highest_pct - second_highest_pct
                max_add_amts = {tkn: ss.liquidity[tkn] * pct_diff for tkn in tkns}
                for j, tkn in enumerate(ss.asset_list):  # add liquidity until right amount of shares are obtained
                    if tkn in max_add_amts:
                        max_add_amt = max_add_amts[tkn]
                        # mint any missing assets to pool_agent
                        mint_to_quantity(pool_agent, tkn, max_add_amt, minted)
                        # try adding liquidity
                        test_state, test_agent = simulate_add_liquidity(ss, pool_agent, max_add_amt, tkn)
                        delta_s = test_agent.holdings[ss.unique_id] - (pool_agent.holdings[ss.unique_id] if ss.unique_id in pool_agent.holdings else 0)
                        if delta_s > stableswap_deltas[i][0]:  # need to buy shares instead
                            init_holdings = pool_agent.holdings[tkn] if tkn in pool_agent.holdings else 0
                            ss.buy_shares(pool_agent, stableswap_deltas[i][0], tkn)
                            delta_tkn = init_holdings - pool_agent.holdings[tkn]
                            stableswap_deltas[i][j+1] -= delta_tkn
                            stableswap_deltas[i][0] -= stableswap_deltas[i][0]
                        else:
                            init_shares = pool_agent.holdings[ss.unique_id] if ss.unique_id in pool_agent.holdings else 0
                            ss.add_liquidity(pool_agent, max_add_amt, tkn)
                            delta_shares = pool_agent.holdings[ss.unique_id] - init_shares
                            stableswap_deltas[i][0] -= delta_shares
                            stableswap_deltas[i][j+1] -= max_add_amt
                        # burn excess minted assets
                        burn_excess(pool_agent, tkn, minted)
        elif stableswap_deltas[i][0] < 0:
            while stableswap_deltas[i][0] < 0:
                # construct list of tokens to remove liquidity in
                if min(stableswap_deltas[i][1:]) >= 0:
                    break  # can't remove any more liquidity
                tkns = []
                highest_pct = 0
                second_highest_pct = 0
                min_pct = float('inf')
                tol = 1e-9
                for j, tkn in enumerate(ss.asset_list):  # calculate highest and second highest pcts
                    pct = -pool_deltas[j + 1] / ss.liquidity[tkn]
                    min_pct = min(min_pct, pct)
                    if pool_deltas[j+1] < 0:
                        rel_diff = (pct - highest_pct)/highest_pct if highest_pct > 0 else float("inf")
                        if rel_diff > tol:
                            second_highest_pct = highest_pct
                            highest_pct = pct
                            tkns = [tkn]
                        elif rel_diff > -tol:
                            tkns.append(tkn)
                if min_pct > 0:  # do a uniform withdrawal of min_pct
                    delta_s = min(min_pct * ss.shares, -stableswap_deltas[i][0])
                    # mint any missing shares to pool_agent
                    mint_to_quantity(pool_agent, ss.unique_id, delta_s, minted)
                    # remove liquidity
                    init_amts = {tkn: pool_agent.holdings[tkn] if tkn in pool_agent.holdings else 0 for tkn in ss.asset_list}
                    ss.remove_uniform(pool_agent, delta_s)
                    stableswap_deltas[i][0] += delta_s
                    for j, tkn in enumerate(ss.asset_list):
                        stableswap_deltas[i][j+1] += pool_agent.holdings[tkn] - init_amts[tkn]
                    # burn excess minted assets
                    burn_excess(pool_agent, ss.unique_id, minted)
                else:  # need to withdraw non-uniformly
                    pct_diff = highest_pct - second_highest_pct
                    max_delta_s = -stableswap_deltas[i][0]
                    max_remove_amts = {tkn: ss.liquidity[tkn] * pct_diff for tkn in tkns}
                    for j, tkn in enumerate(ss.asset_list):  # remove liquidity until right amount of shares are burned
                        if tkn in max_remove_amts:
                            max_remove_amt = max_remove_amts[tkn]
                            max_shares_burn = min(max_delta_s / len(tkns), -stableswap_deltas[i][0])
                            # mint any missing shares to pool_agent
                            mint_to_quantity(pool_agent, ss.unique_id, max_shares_burn, minted)
                            # try removing liquidity
                            test_state, test_agent = simulate_remove_liquidity(ss, pool_agent, max_shares_burn, tkn)
                            removed_amt = test_agent.holdings[tkn] - (pool_agent.holdings[tkn] if tkn in pool_agent.holdings else 0)
                            if removed_amt > max_remove_amt:  # removed too much tkn
                                # do withdraw asset instead
                                init_holdings = pool_agent.holdings[ss.unique_id] if ss.unique_id in pool_agent.holdings else 0
                                ss.withdraw_asset(pool_agent, max_remove_amt, tkn)
                                delta_s = init_holdings - pool_agent.holdings[ss.unique_id]
                                stableswap_deltas[i][j+1] += max_remove_amt
                                stableswap_deltas[i][0] += delta_s
                            else:
                                init_holdings = pool_agent.holdings[tkn] if tkn in pool_agent.holdings else 0
                                ss.remove_liquidity(pool_agent, max_shares_burn, tkn)
                                delta_tkn = pool_agent.holdings[tkn] - init_holdings
                                stableswap_deltas[i][0] += max_shares_burn
                                stableswap_deltas[i][j+1] += delta_tkn
                            # burn excess minted assets
                            burn_excess(pool_agent, ss.unique_id, minted)

        # do trades in stableswap pool
        for j, tkn_buy in enumerate(ss.asset_list):
            if stableswap_deltas[i][j+1] < 0:
                for k, tkn_sell in enumerate(ss.asset_list):
                    if stableswap_deltas[i][k+1] > 0:
                        max_sell_amt = stableswap_deltas[i][k+1]
                        max_buy_amt = -stableswap_deltas[i][j+1]
                        # mint any missing assets to pool_agent
                        mint_to_quantity(pool_agent, tkn_sell, max_sell_amt, minted)
                        # try to sell tkn_sell for tkn_buy
                        test_state, test_agent = simulate_swap(ss, pool_agent, tkn_buy, tkn_sell, sell_quantity=max_sell_amt)
                        buy_given_max_sell = test_agent.holdings[tkn_buy] - (pool_agent.holdings[tkn_buy] if tkn_buy in pool_agent.holdings else 0)
                        if buy_given_max_sell > max_buy_amt:
                            init_sell_holdings = pool_agent.holdings[tkn_sell]
                            ss.swap(pool_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, buy_quantity=max_buy_amt)
                            stableswap_deltas[i][j+1] += max_buy_amt
                            stableswap_deltas[i][k+1] -= init_sell_holdings - pool_agent.holdings[tkn_sell]
                        else:
                            init_buy_holdings = pool_agent.holdings[tkn_buy] if tkn_buy in pool_agent.holdings else 0
                            ss.swap(pool_agent, tkn_buy=tkn_buy, tkn_sell=tkn_sell, sell_quantity=max_sell_amt)
                            stableswap_deltas[i][k+1] -= max_sell_amt
                            stableswap_deltas[i][j+1] += pool_agent.holdings[tkn_buy] - init_buy_holdings
                        # burn excess minted assets
                        burn_excess(pool_agent, tkn_sell, minted)

    # Omnipool trades
    init_lrna = {tkn: omnipool.lrna[tkn] for tkn in omnipool_deltas if omnipool_deltas[tkn] != 0}
    for tkn_buy in omnipool_deltas:
        if omnipool_deltas[tkn_buy] < 0:
            for tkn_sell in omnipool_deltas:
                # try to sell tkn_buy for tkn_sell
                if omnipool_deltas[tkn_sell] > 0:
                    max_buy_amt = -omnipool_deltas[tkn_buy]
                    max_buy_amt = math.nextafter(max_buy_amt, math.inf) if max_buy_amt != 0 else 0
                    max_sell_amt = omnipool_deltas[tkn_sell]
                    max_sell_amt = math.nextafter(max_sell_amt, -math.inf) if max_sell_amt != 0 else 0
                    # mint any missing assets to pool_agent
                    mint_to_quantity(pool_agent, tkn_sell, max_sell_amt, minted)
                    test_state, test_agent = simulate_swap(omnipool, pool_agent, tkn_buy, tkn_sell, sell_quantity=max_sell_amt)
                    buy_given_max_sell = test_agent.holdings[tkn_buy] - (pool_agent.holdings[tkn_buy] if tkn_buy in pool_agent.holdings else 0)
                    if buy_given_max_sell > max_buy_amt:  # can't do max sell, do max buy instead
                        init_sell_holdings = pool_agent.holdings[tkn_sell]
                        omnipool.swap(pool_agent, tkn_buy, tkn_sell, buy_quantity=max_buy_amt)
                        omnipool_deltas[tkn_buy] += max_buy_amt
                        omnipool_deltas[tkn_sell] -= init_sell_holdings - pool_agent.holdings[tkn_sell]
                    else:
                        init_buy_holdings = pool_agent.holdings[tkn_buy] if tkn_buy in pool_agent.holdings else 0
                        omnipool.swap(pool_agent, tkn_buy, tkn_sell, sell_quantity=max_sell_amt)
                        omnipool_deltas[tkn_sell] -= max_sell_amt
                        omnipool_deltas[tkn_buy] += pool_agent.holdings[tkn_buy] - init_buy_holdings
                    # burn excess tkn_sell
                    burn_excess(pool_agent, tkn_sell, minted)
        if tkn_buy in deltas:
            # transfer matched fees to fee agent
            matched_amt = min(deltas[tkn_buy]["in"], deltas[tkn_buy]["out"])
            fee_amt = matched_amt * fee_match
            if fee_amt > 0:
                pool_agent.holdings[tkn_buy] -= fee_amt
                fee_agent.holdings[tkn_buy] = fee_amt

    lrna_deltas = {tkn: omnipool.lrna[tkn] - init_lrna[tkn] for tkn in init_lrna}

    # transfer assets in from agents whose intents are being executed
    for transfer in transfers:
        if transfer['sell_quantity'] > 0:
            if transfer['tkn_sell'] not in pool_agent.holdings:
                pool_agent.holdings[transfer['tkn_sell']] = 0
            pool_agent.holdings[transfer['tkn_sell']] += transfer['sell_quantity']
            transfer['agent'].holdings[transfer['tkn_sell']] -= transfer['sell_quantity']
        elif transfer['sell_quantity'] < 0:
            raise Exception("sell quantity is negative")

    # transfer assets out to intent agents
    for transfer in transfers:
        if transfer['buy_quantity'] > 0:
            pool_agent.holdings[transfer['tkn_buy']] -= transfer['buy_quantity']
            if transfer['tkn_buy'] not in transfer['agent'].holdings:
                transfer['agent'].holdings[transfer['tkn_buy']] = 0
            transfer['agent'].holdings[transfer['tkn_buy']] += transfer['buy_quantity']
        elif transfer['buy_quantity'] < 0:
            raise Exception("buy quantity is negative")

    for tkn in minted:
        pool_agent.holdings[tkn] -= minted[tkn]
        minted[tkn] = 0

    if exit_to is not None:  # have pool agent exit to a specific token
        tkns_stuck = []
        tkns_to_sell = [tkn for tkn in pool_agent.holdings if (tkn != exit_to and pool_agent.holdings[tkn] > 0)]
        while len(tkns_to_sell) > 0:  # first, sell extra tokens
            tkn = tkns_to_sell.pop()
            if pool_agent.holdings[tkn] > 0:
                if tkn in omnipool.asset_list:
                    omnipool.swap(pool_agent, exit_to, tkn, sell_quantity=pool_agent.holdings[tkn])
                    if pool_agent.holdings[tkn] > 0:
                        tkns_stuck.append(tkn)
                else:
                    for ss in stableswap_list:
                        if tkn == ss.unique_id:  # withdraw assets uniformly
                            ss.remove_uniform(pool_agent, pool_agent.holdings[tkn])
                            for tkn_bought in ss.asset_list:
                                tkns_to_sell.append(tkn_bought)
                        elif tkn in ss.asset_list:  # sell asset for something in Omnipool
                            if ss.unique_id in omnipool.asset_list:
                                ss.add_liquidity(pool_agent, pool_agent.holdings[tkn], tkn)
                                tkns_to_sell.append(ss.unique_id)
                            else:
                                for tkn_buy in ss.asset_list:
                                    if tkn_buy in omnipool.asset_list:
                                        ss.swap(pool_agent, tkn_buy=tkn_buy, tkn_sell=tkn, sell_quantity=pool_agent.holdings[tkn])
                                        if pool_agent.holdings[tkn] > 0:
                                            tkns_stuck.append(tkn)
                                        else:
                                            tkns_to_sell.append(tkn_buy)
                                    else:
                                        tkns_stuck.append(tkn)

        tkns_stuck = []
        tkns_to_buy = [tkn for tkn in pool_agent.holdings if pool_agent.holdings[tkn] < 0]
        while len(tkns_to_buy) > 0:  # then, buy missing tokens
            tkn = tkns_to_buy.pop()
            if pool_agent.holdings[tkn] < 0:
                if tkn in omnipool.asset_list:
                    omnipool.swap(pool_agent, tkn, exit_to, buy_quantity=-pool_agent.holdings[tkn])
                    if pool_agent.holdings[tkn] < 0:
                        tkns_stuck.append(tkn)
                else:
                    for ss in stableswap_list:
                        if tkn == ss.unique_id:
                            tkns_stuck.append(tkn)  # for now we will only worry about stablepools with share tokens in Omnipool
                        elif tkn in ss.asset_list:
                            if ss.unique_id in omnipool.asset_list:
                                # mint some shares to pool_agent
                                share_quantity = abs(pool_agent.holdings[tkn]) * ss.shares / ss.liquidity[tkn]
                                mint_to_quantity(pool_agent, ss.unique_id, share_quantity, minted)
                                ss.withdraw_asset(pool_agent, -pool_agent.holdings[tkn], tkn)
                                # burn minted shares
                                pool_agent.holdings[ss.unique_id] -= minted[ss.unique_id]
                                minted[ss.unique_id] = 0
                                tkns_to_buy.append(ss.unique_id)
                                break
                            else:
                                tkns_stuck.append(tkn)  # for now we assume ss.unique_id is in Omnipool


    return pool_agent, fee_agent, lrna_deltas


def validate_remainder(pool_agent: Agent):
    for tkn, amt in pool_agent.holdings.items():
        if amt < -1e-10:
            # return False
            raise
    return True


def update_intents(intents: list, transfers: list):
    for i in range(len(intents)):
        intent, transfer = intents[i], transfers[i]
        intent['sell_quantity'] -= transfer['sell_quantity']
        intent['buy_quantity'] -= transfer['buy_quantity']
