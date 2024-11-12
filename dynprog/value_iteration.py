import numpy as np


def get_income_mean():
    return 1


def get_income_dispersion():
    return 0.3


def get_beta():
    return 0.92


def get_interest_rate():
    return 0.03


def get_risk_aversion_coefficient():
    return 2


def get_initial_assets():
    return 1


def get_state_list():
    state_ls = ["H", "L"]
    return state_ls


def get_probability(state):
    if state == "H":
        probability_dd = {
            "H": 0.5,
            "L": 0.5,
        }
    elif state == "L":
        probability_dd = {
            "H": 0.5,
            "L": 0.5,
        }
    else:
        msg = f"state {state} not implemented!"
        raise Exception(msg)
    return probability_dd


def calculate_utility(consumption):
    theta = get_risk_aversion_coefficient()
    utility = consumption ** (1 - theta) / (1 - theta)
    return utility


def get_income(state):
    income_mean = get_income_mean()
    income_dispersion = get_income_dispersion()
    if state == "H":
        return income_mean + income_dispersion
    elif state == "L":
        return income_mean - income_dispersion
    else:
        msg = f"state {state} not implemented!"
        raise Exception(msg)


def get_cash_on_hand(state, assets):
    income = get_income(state=state)
    return income + assets


def calculate_consumption(savings, interest_rate, cash_on_hand):
    consumption = cash_on_hand - savings / (1 + interest_rate)
    return consumption


def calculate_natural_borrowing_limit():
    income_mean = get_income_mean()
    income_dispersion = get_income_dispersion()
    income_min = income_mean - income_dispersion

    interest_rate = get_interest_rate()
    natural_borrowing_limit = - (1 + interest_rate) / interest_rate * income_min
    return natural_borrowing_limit


def create_grid(lower_bound, upper_bound=None, length=100):
    if upper_bound is None:
        if lower_bound < 0:
            upper_bound = - lower_bound
        else:
            upper_bound = 2 * lower_bound

    diff = (upper_bound - lower_bound) / (length - 1)
    range_ls = list(range(length))
    grid_ls = [lower_bound + i * diff for i in range_ls]
    return grid_ls


def get_expected_value(value_next_dd, state, savings):
    probability_dd = get_probability(state=state)
    expected = 0
    for next_state, state_value_dd in value_next_dd.items():
        next_state_probability = probability_dd.get(next_state, 0)
        next_state_value = state_value_dd.get(savings, 0)
        expected += next_state_probability * next_state_value
    return expected


def calculate_period_value(savings, interest_rate, cash_on_hand, value_next_dd, state, beta):
    consumption = calculate_consumption(savings=savings, interest_rate=interest_rate, cash_on_hand=cash_on_hand)
    period_utility = calculate_utility(consumption=consumption)
    next_period_value = get_expected_value(value_next_dd=value_next_dd, state=state, savings=savings)
    value = period_utility + beta * next_period_value
    return value


def calculate_distance(state_dd, prev_dd):
    value_key_ls = list(state_dd.keys())
    prev_key_ls = list(prev_dd.keys())
    key_ls = sorted([i for i in value_key_ls if i in prev_key_ls])
    distance_ls = [abs(state_dd.get(key, 0) - prev_dd.get(key, 0)) for key in key_ls]
    distance = max(distance_ls)
    return distance
