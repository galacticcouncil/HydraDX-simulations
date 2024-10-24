import copy
from pprint import pprint

from hypothesis import given, strategies as st, assume, settings, Verbosity

from hydradx.model.amm.agents import Agent
from hydradx.model.amm.omnipool_amm import OmnipoolState
from mpmath import mp, mpf

from hydradx.model.amm.omnix import validate_and_execute_solution
from hydradx.model.amm.omnix_solver_simple import find_solution, find_solution2, find_solution3, \
    _find_solution_unrounded3, add_buy_deltas, round_solution, find_solution_outer_approx, _solve_inclusion_problem


def test_single_trade_settles():
    agents = [Agent(holdings={'DOT': 100, 'LRNA': 750})]

    init_intents_partial = [  # selling DOT for $7
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True}
    ]
    init_intents_full = [  # selling DOT for $7
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False}
    ]
    init_intents_partial_lrna = [
        {'sell_quantity': mpf(750), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': True}
    ]
    init_intents_full_lrna = [
        {'sell_quantity': mpf(750), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': False}
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intents = copy.deepcopy(init_intents_partial)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == -init_intents_partial[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_partial[0]['buy_quantity']

    intents = copy.deepcopy(init_intents_full)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == -init_intents_full[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_full[0]['buy_quantity']

    intents = copy.deepcopy(init_intents_partial_lrna)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == -init_intents_partial_lrna[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_partial_lrna[0]['buy_quantity']

    intents = copy.deepcopy(init_intents_full_lrna)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == -init_intents_full_lrna[0]['sell_quantity']
    assert intent_deltas[0][1] == init_intents_full_lrna[0]['buy_quantity']


def test_single_trade_does_not_settle():
    agents = [Agent(holdings={'DOT': 100, 'USDT': 0})]

    init_intents_partial = [  # selling DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(800), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True}
    ]
    init_intents_full = [  # selling DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(800), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False}
    ]
    init_intents_partial_lrna = [
        {'sell_quantity': mpf(650), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': True}
    ]
    init_intents_full_lrna = [
        {'sell_quantity': mpf(650), 'buy_quantity': mpf(700), 'tkn_sell': 'LRNA', 'tkn_buy': 'USDT', 'agent': agents[0],
         'partial': False}
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intents = copy.deepcopy(init_intents_partial)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_full)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_partial_lrna)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0

    intents = copy.deepcopy(init_intents_full_lrna)
    intent_deltas = find_solution_outer_approx(initial_state, intents)
    assert validate_and_execute_solution(initial_state.copy(), intents, intent_deltas)
    assert intent_deltas[0][0] == 0
    assert intent_deltas[0][1] == 0


def test_matching_trades_execute_more():
    agents = [Agent(holdings={'DOT': 1000}), Agent(holdings={'USDT': 7600})]

    intent1 = {  # selling DOT for $7.49
        'sell_quantity': mpf(1000), 'buy_quantity': mpf(7470), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True
    }

    intent2 = {  # buying DOT for $7.51
        'sell_quantity': mpf(7530), 'buy_quantity': mpf(1000), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[1], 'partial': True
    }

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}  # DOT price is $7.50
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    # do the DOT sale alone
    state_sale = initial_state.copy()
    intents_sale = [copy.deepcopy(intent1)]
    sale_deltas = find_solution_outer_approx(state_sale, intents_sale)
    assert validate_and_execute_solution(state_sale, intents_sale, sale_deltas)

    # do the DOT buy alone
    state_buy = initial_state.copy()
    intents_buy = [copy.deepcopy(intent2)]
    buy_deltas = find_solution_outer_approx(state_buy, intents_buy)
    assert validate_and_execute_solution(state_buy, intents_buy, buy_deltas)

    # do both trades together
    state_match = initial_state.copy()
    intents_match = [copy.deepcopy(intent1), copy.deepcopy(intent2)]
    match_deltas = find_solution_outer_approx(state_match, intents_match)
    assert validate_and_execute_solution(state_match, intents_match, match_deltas)

    # check that matching trades caused more execution than executing either alone
    assert abs(match_deltas[0][0]) > abs(sale_deltas[0][0])
    assert abs(match_deltas[1][0]) > abs(buy_deltas[0][0])


def test_convex():

    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3], 'partial': True},  # selling HDX for $1
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intent_deltas = find_solution_outer_approx(initial_state, intents)

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)


def test_with_lrna_intent():
    agents = [
        Agent(holdings={'DOT': 100}),
        Agent(holdings={'USDT': 1500}),
        Agent(holdings={'USDT': 400}),
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'LRNA': 1000}),
        Agent(holdings={'DOT': 1000000})
    ]

    intents = [
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': True},  # selling DOT for $7
        {'sell_quantity': mpf(1500), 'buy_quantity': mpf(100000), 'tkn_sell': 'USDT', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},  # buying HDX for $0.015
        {'sell_quantity': mpf(400), 'buy_quantity': mpf(50), 'tkn_sell': 'USDT', 'tkn_buy': 'DOT', 'agent': agents[2], 'partial': True},  # buying DOT for $8
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(100), 'tkn_sell': 'HDX', 'tkn_buy': 'USDT', 'agent': agents[3], 'partial': True},  # selling HDX for $1
        {'sell_quantity': mpf(1000), 'buy_quantity': mpf(100), 'tkn_sell': 'LRNA', 'tkn_buy': 'DOT', 'agent': agents[4], 'partial': True},  # buying DOT for $10
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(700), 'tkn_sell': 'DOT', 'tkn_buy': 'USDT', 'agent': agents[0], 'partial': False},  # selling DOT for $7
    ]

    liquidity = {'HDX': mpf(100000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000/7.5)}
    lrna = {'HDX': mpf(1000000), 'USDT': mpf(10000000), 'DOT': mpf(10000000)}
    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intent_deltas = find_solution_outer_approx(initial_state, intents)

    assert validate_and_execute_solution(initial_state, intents, intent_deltas)

    pprint(intent_deltas)


def test_solver_with_real_omnipool():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'CRU': 5}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        # {'sell_quantity': mpf(1.149711278057), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1]},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': True},
        {'sell_quantity': mpf(1.150), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intent_deltas = find_solution_outer_approx(initial_state, intents)

    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)

    pprint(intent_deltas)


def test_solver_with_real_omnipool_one_full():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': False},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
         'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    partial_intents = [i for i in intents if i['partial']]
    full_intents = [i for i in intents if not i['partial']]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    full_intent_indicators = [1]
    # full_intent_indicators = []

    amm_deltas, sell_deltas, _, _ = _find_solution_unrounded3(initial_state, partial_intents, full_intents, I=full_intent_indicators)
    for i in full_intents:
        if full_intent_indicators.pop(0) == 1:
            sell_deltas.append(-i['sell_quantity'])

    sell_deltas = round_solution(partial_intents + full_intents, sell_deltas)
    intent_deltas = add_buy_deltas(partial_intents + full_intents, sell_deltas)


    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(partial_intents + full_intents), intent_deltas)

    pprint(intent_deltas)


def test_full_solver():
    agents = [
        Agent(holdings={'HDX': 100}),
        Agent(holdings={'HDX': 100}),
    ]

    intents = [
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149711278057), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0]},
        # {'sell_quantity': mpf(1.149711278057), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX', 'agent': agents[1]},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[0], 'partial': False},
        {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.149), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1], 'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(200.0), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU', 'agent': agents[1],
        #  'partial': True},
        # {'sell_quantity': mpf(100), 'buy_quantity': mpf(1.25359), 'tkn_sell': 'HDX', 'tkn_buy': 'CRU',
        #  'agent': agents[0]},
        # {'sell_quantity': mpf(1.25361), 'buy_quantity': mpf(100), 'tkn_sell': 'CRU', 'tkn_buy': 'HDX',
        #  'agent': agents[1]}
    ]

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    intent_deltas = find_solution_outer_approx(initial_state, intents)

    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)

    pprint(intent_deltas)

@given(st.lists(st.floats(min_value=1e-10, max_value=0.5), min_size=3, max_size=3),
       st.lists(st.floats(min_value=0.9, max_value=1.1), min_size=3, max_size=3),
        st.lists(st.integers(min_value=0, max_value=18), min_size=3, max_size=3),
        st.lists(st.integers(min_value=0, max_value=17), min_size=3, max_size=3),
        st.lists(st.booleans(), min_size=3, max_size=3)
       )
@settings(print_blob=True, verbosity=Verbosity.verbose, deadline=None)
def test_solver_random_intents(sell_ratios, price_ratios, sell_is, buy_is, partial_flags):

    liquidity = {'4-Pool': mpf(1392263.9295618401), 'HDX': mpf(140474254.46393022), 'KILT': mpf(1941765.8700688032),
                 'WETH': mpf(897.820372708098), '2-Pool-btc': mpf(80.37640742108785), 'GLMR': mpf(7389788.325282889),
                 'BNC': mpf(5294190.655262755), 'RING': mpf(30608622.54045291), 'vASTR': mpf(1709768.9093601815),
                 'vDOT': mpf(851755.7840315843), 'CFG': mpf(3497639.0397717496), 'CRU': mpf(337868.26827475097),
                 '2-Pool': mpf(14626788.977583803), 'DOT': mpf(2369965.4990946855), 'PHA': mpf(6002455.470581388),
                 'ZTG': mpf(9707643.829161936), 'INTR': mpf(52756928.48950746), 'ASTR': mpf(31837859.71273387), }
    lrna = {'4-Pool': mpf(50483.454258911326), 'HDX': mpf(24725.8021660851), 'KILT': mpf(10802.301353604526),
            'WETH': mpf(82979.9927924809), '2-Pool-btc': mpf(197326.54331209575), 'GLMR': mpf(44400.11377262768),
            'BNC': mpf(35968.10763198863), 'RING': mpf(1996.48438233777), 'vASTR': mpf(4292.819030020081),
            'vDOT': mpf(182410.99000727307), 'CFG': mpf(41595.57689216696), 'CRU': mpf(4744.442135139952),
            '2-Pool': mpf(523282.70722423657), 'DOT': mpf(363516.4838824808), 'PHA': mpf(24099.247547699764),
            'ZTG': mpf(4208.90365804613), 'INTR': mpf(19516.483401186168), 'ASTR': mpf(68571.5237579274), }

    initial_state = OmnipoolState(
        tokens={
            tkn: {'liquidity': liquidity[tkn], 'LRNA': lrna[tkn]} for tkn in lrna
        },
        asset_fee=mpf(0.0025),
        lrna_fee=mpf(0.0005)
    )
    initial_state.last_fee = {tkn: mpf(0.0025) for tkn in lrna}
    initial_state.last_lrna_fee = {tkn: mpf(0.0005) for tkn in lrna}

    good_indices = [i for i in range(len(sell_is)) if sell_is[i]-1 != buy_is[i]]
    intents = []
    for i in good_indices:
        sell_tkn = initial_state.asset_list[sell_is[i]-1] if sell_is[i] > 0 else "LRNA"
        buy_tkn = initial_state.asset_list[buy_is[i]]
        if sell_tkn != "LRNA":
            sell_quantity = sell_ratios[i] * liquidity[sell_tkn]
        else:
            sell_quantity = sell_ratios[i] * lrna[buy_tkn]
        buy_quantity = sell_quantity * initial_state.price(initial_state, sell_tkn, buy_tkn) * price_ratios[i]
        agent = Agent(holdings={sell_tkn: sell_quantity})
        intents.append({'sell_quantity': sell_quantity, 'buy_quantity': buy_quantity, 'tkn_sell': sell_tkn,
                        'tkn_buy': buy_tkn, 'agent': agent, 'partial': partial_flags[i]})

    intent_deltas = find_solution_outer_approx(initial_state, intents)

    assert validate_and_execute_solution(initial_state.copy(), copy.deepcopy(intents), intent_deltas)

    pprint(intent_deltas)
