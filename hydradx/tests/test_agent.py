import pytest

from hydradx.model.amm.agents import Agent


def test_validate_holdings():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    if agent.validate_holdings('USDT') != True:
        raise
    if agent.validate_holdings('DOT') != False:
        raise
    if agent.validate_holdings('ETH') != False:
        raise
    if agent.validate_holdings('USDT', 50) != True:
        raise
    if agent.validate_holdings('USDT', 100) != True:
        raise
    if agent.validate_holdings('USDT', 101) != False:
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


def test_add():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    agent.add('USDT', 50)
    if agent.get_holdings('USDT') != 150:
        raise
    agent.add('DOT', 50)
    if agent.get_holdings('DOT') != 50:
        raise
    agent.add('ETH', 50)
    if agent.get_holdings('ETH') != 50:
        raise


def test_remove():
    holdings = {'USDT': 100, 'DOT': 0}
    agent = Agent(holdings=holdings)
    agent.remove('USDT', 50)
    if agent.get_holdings('USDT') != 50:
        raise
    agent.remove('USDT', 50)
    if agent.get_holdings('USDT') != 0:
        raise
    agent.remove('USDT', 50)
    if agent.get_holdings('USDT') != 0:
        raise
    agent.remove('DOT', 50)
    if agent.get_holdings('DOT') != 0:
        raise
    agent.remove('ETH', 50)
    if 'ETH' in agent.holdings:
        raise

    test_agent = Agent(holdings=holdings, enforce_holdings=False)
    test_agent.remove('DOT', 50)
    if test_agent.get_holdings('DOT') != -50:
        raise
    test_agent.remove('ETH', 50)
    if test_agent.get_holdings('ETH') != -50:
        raise
