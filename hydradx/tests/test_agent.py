import pytest

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


def test_get_holdings():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    if agent.get_holdings('USDT') != 100:
        raise
    if agent.get_holdings('DOT') != 0:
        raise
    if agent.get_holdings('ETH') != 0:
        raise


def test_transfer_to():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    agent.transfer_to('USDT', 50)
    if agent.get_holdings('USDT') != 150:
        raise
    agent.transfer_to('DOT', 50)
    if agent.get_holdings('DOT') != 50:
        raise
    agent.transfer_to('ETH', 50)
    if agent.get_holdings('ETH') != 50:
        raise


def test_transfer_from():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    agent.transfer_from('USDT', 50)
    if agent.get_holdings('USDT') != 50:
        raise
    agent.transfer_from('USDT', 50)
    if agent.get_holdings('USDT') != 0:
        raise
    with pytest.raises(ValueError):
        agent.transfer_from('USDT', 50)
    with pytest.raises(ValueError):
        agent.transfer_from('DOT', 50)
    agent.transfer_from('DOT', 50, enforce_holdings=False)
    if agent.get_holdings('DOT') != -50:
        raise
    agent.transfer_from('ETH', 50, enforce_holdings=False)
    if agent.get_holdings('ETH') != -50:
        raise
