from hydradx.model.amm.agents import Agent


def test_is_holding():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    if agent.is_holding('USDT') != True:
        raise
    if agent.is_holding('DOT') != False:
        raise
    if agent.is_holding('ETH') != False:
        raise
    if agent.is_holding('USDT', 50) != True:
        raise
    if agent.is_holding('USDT', 100) != True:
        raise
    if agent.is_holding('USDT', 101) != False:
        raise
