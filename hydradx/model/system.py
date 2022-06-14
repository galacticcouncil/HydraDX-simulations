from .amm import amm


def execute_trades(params, substep, state_history, prev_state, policy_input):
    market = prev_state['state']['amm']
    agents = prev_state['state']['agents']
    for agent_id, agent in agents.items():
        if 'trade_strategy' in agent and agent['trade_strategy']:
            market, agents = agent['trade_strategy'].execute(agents=agents, agent_id=agent_id, market=market)
    return 'state', {'amm': market, 'agents': agents, 'external': prev_state['state']['external']}
