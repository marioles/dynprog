import math
import numpy as np
import pandas as pd
import warnings

from typing import Optional


class ValueIterator(object):
    # required parameters
    _beta = None
    _interest_rate = None

    # utility function
    _theta = None  # for CRRA utility

    # income
    _income_mean = None
    _income_dispersion = None

    # initial conditions
    _initial_assets = None
    _natural_borrowing_limit = None

    # states
    _state_ls = list()
    _probability_dd = dict()

    # optimization
    _grid_spacing = 0.05
    _cash_on_hand_ls = list()
    _value_next_dd = dict()
    _max_iterations = 10000
    _epsilon = 1e-5

    def __init__(self, **kwargs):
        self.set_parameters(**kwargs)

    def _set_beta(self, beta: float) -> None:
        self._beta = beta

    def _set_interest_rate(self, interest_rate: float) -> None:
        self._interest_rate = interest_rate

    def _set_theta(self, theta: float) -> None:
        self._theta = theta

    def _set_income_mean(self, income_mean: float) -> None:
        self._income_mean = income_mean

    def _set_income_dispersion(self, income_dispersion: float) -> None:
        self._income_dispersion = income_dispersion

    def _set_initial_assets(self, initial_assets: float) -> None:
        self._initial_assets = initial_assets

    def _set_state_list(self, state_ls: list) -> None:
        self._state_ls = state_ls

    def _set_probability_dict(self, probability_dd: dict) -> None:
        self._probability_dd = probability_dd

    def _set_grid_spacing(self, grid_spacing: int) -> None:
        self._grid_spacing = grid_spacing

    def _set_max_iterations(self, max_iterations: int) -> None:
        self._max_iterations = max_iterations

    def _set_epsilon(self, epsilon: int) -> None:
        self._epsilon = epsilon

    def set_parameters(self, **kwargs) -> None:
        for param, value in kwargs.items():
            if param == "beta":
                self._set_beta(beta=value)
            elif param == "interest_rate":
                self._set_interest_rate(interest_rate=value)
            elif param == "theta":
                self._set_theta(theta=value)
            elif param == "income_mean":
                self._set_income_mean(income_mean=value)
            elif param == "income_dispersion":
                self._set_income_dispersion(income_dispersion=value)
            elif param == "theta":
                self._set_theta(theta=value)
            elif param == "initial_assets":
                self._set_initial_assets(initial_assets=value)
            elif param == "state_ls":
                self._set_state_list(state_ls=value)
            elif param == "probability_dd":
                self._set_probability_dict(probability_dd=value)
            elif param == "grid_spacing":
                self._set_grid_spacing(grid_spacing=value)
            elif param == "max_iterations":
                self._set_max_iterations(max_iterations=value)
            elif param == "epsilon":
                self._set_epsilon(epsilon=value)
            else:
                msg = f"parameter not recognized: {param}"
                warnings.warn(message=msg)

    def get_param(self, param: str) -> [float, list, dict]:
        if param == "beta":
            value = self._beta
        elif param == "interest_rate":
            value = self._interest_rate
        elif param == "theta":
            value = self._theta
        elif param == "income_mean":
            value = self._income_mean
        elif param == "income_dispersion":
            value = self._income_dispersion
        elif param == "theta":
            value = self._theta
        elif param == "initial_assets":
            value = self._initial_assets
        elif param == "state_ls":
            value = self._state_ls
        elif param == "probability_dd":
            value = self._probability_dd
        elif param == "grid_spacing":
            value = self._grid_spacing
        elif param == "cash_on_hand_ls":
            value = self._cash_on_hand_ls
        elif param == "value_next_dd":
            value = self._value_next_dd
        elif param == "natural_borrowing_limit":
            value = self._natural_borrowing_limit
        elif param == "epsilon":
            value = self._epsilon
        elif param == "max_iterations":
            value = self._max_iterations
        else:
            msg = f"parameter not recognized: {param}"
            warnings.warn(message=msg)
            value = None
        return value

    def _get_income(self, state: str) -> float:
        income_mean = self.get_param(param="income_mean")
        income_dispersion = self.get_param(param="income_dispersion")

        if state == "H":
            return income_mean + income_dispersion
        elif state == "L":
            return income_mean - income_dispersion
        else:
            msg = f"state {state} not implemented!"
            raise Exception(msg)

    def _calculate_natural_borrowing_limit(self) -> float:
        income_min = self._get_income(state="L")

        interest_rate = self.get_param(param="interest_rate")
        natural_borrowing_limit = - (1 + interest_rate) / interest_rate * income_min
        return natural_borrowing_limit

    def _set_natural_borrowing_limit(self) -> None:
        natural_borrowing_limit = self._calculate_natural_borrowing_limit()
        self._natural_borrowing_limit = natural_borrowing_limit

    def _construct_cash_on_hand_grid(self) -> list:
        grid_spacing = self.get_param(param="grid_spacing")

        self._set_natural_borrowing_limit()
        natural_borrowing_limit = self.get_param(param="natural_borrowing_limit")
        income_min = self._get_income(state="L")
        income_max = self._get_income(state="H")

        lower_bound = income_min + natural_borrowing_limit
        upper_bound = income_max - natural_borrowing_limit

        second_lower_bound = math.ceil(int(lower_bound / grid_spacing)) * grid_spacing
        second_upper_bound = math.floor(int(upper_bound / grid_spacing)) * grid_spacing

        bins = int((second_upper_bound - second_lower_bound) / grid_spacing)
        cash_on_hand_ls = [second_lower_bound + i * grid_spacing for i in range(bins + 1)]
        cash_on_hand_ls = [round(i, 3) for i in cash_on_hand_ls]

        return cash_on_hand_ls

    def _set_cash_on_hand_grid(self):
        cash_on_hand_ls = self._construct_cash_on_hand_grid()
        self._cash_on_hand_ls = cash_on_hand_ls

    def _construct_value_dict(self, cash_on_hand_ls: list, fill_value: Optional[float] = None) -> dict:
        state_ls = self.get_param(param="state_ls")
        fill_dd = {
            "consumption": fill_value,
            "savings": fill_value,
            "value": fill_value,
        }
        grid_dd = {i: fill_dd for i in cash_on_hand_ls}
        value_dd = {state: grid_dd.copy() for state in state_ls}
        return value_dd

    def _set_value_dict(self, value_next_dd: Optional[dict] = None) -> None:
        if value_next_dd is None:
            value_next_dd = self.get_param(param="value_next_dd")
            if not value_next_dd:
                self._set_cash_on_hand_grid()
                cash_on_hand_ls = self.get_param(param="cash_on_hand_ls")
                value_next_dd = self._construct_value_dict(cash_on_hand_ls=cash_on_hand_ls, fill_value=0)
        self._value_next_dd = value_next_dd

    def _calculate_consumption(self, cash_on_hand: float, savings: float) -> float:
        interest_rate = self.get_param(param="interest_rate")
        consumption = cash_on_hand - savings / (1 + interest_rate)
        return consumption

    def _calculate_utility(self, consumption: float) -> float:
        theta = self.get_param(param="theta")
        utility = consumption ** (1 - theta) / (1 - theta)
        return utility

    def _calculate_expected_utility(self, savings: float, prev_state: str) -> float:
        state_ls = self.get_param(param="state_ls")
        probability_dd = self.get_param(param="probability_dd")
        value_next_dd = self.get_param(param="value_next_dd")
        expected_utility = 0
        for state in state_ls:
            probability = probability_dd[prev_state][state]
            income = self._get_income(state=state)
            cash_on_hand = round(income + savings, 3)
            state_value = value_next_dd[state][cash_on_hand]["value"]
            expected_utility += probability * state_value
        return expected_utility

    def _calculate_state_value(self, cash_on_hand: float, state: str, income: float) -> dict:
        beta = self.get_param(param="beta")
        natural_borrowing_limit = self.get_param(param="natural_borrowing_limit")

        savings = cash_on_hand - income
        if abs(savings) > abs(natural_borrowing_limit):
            consumption = np.NaN
            value = np.NaN
        else:
            consumption = self._calculate_consumption(cash_on_hand=cash_on_hand, savings=savings)
            utility = self._calculate_utility(consumption=consumption)
            expected = self._calculate_expected_utility(savings=savings, prev_state=state)
            value = utility + beta * expected

        output_dd = {
            "consumption": consumption,
            "savings": savings,
            "value": value,
        }
        return output_dd

    def _calculate_period_value(self) -> dict:
        state_ls = self.get_param(param="state_ls")
        value_dd = dict()
        for state in state_ls:
            income = self._get_income(state=state)
            cash_on_hand_ls = self.get_param(param="cash_on_hand_ls")
            state_value_dd = {
                x: self._calculate_state_value(cash_on_hand=x, state=state, income=income) for x in cash_on_hand_ls
            }
            value_dd[state] = state_value_dd.copy()
        return value_dd

    def _calculate_distance(self, period_dd: dict, prev_dd: dict) -> float:
        period_df = self._convert_dict_to_frame(dd=period_dd, key="period", filter_key="value")
        prev_df = self._convert_dict_to_frame(dd=prev_dd, key="prev", filter_key="value")
        value_df = pd.concat([period_df, prev_df], axis="columns")
        value_df = value_df.dropna(axis="index")

        state_ls = self.get_param(param="state_ls")
        distance_ls = list()
        for state in state_ls:
            state_df = value_df.xs(key=state, axis="columns", level=1)
            diff_ss = state_df["period"] - state_df["prev"]
            abs_ss = diff_ss.abs()
            max_distance = abs_ss.max()
            distance_ls.append(max_distance)
        distance = sum(distance_ls) / len(distance_ls)
        return distance

    @staticmethod
    def _filter_dict(dd: dict, filter_key: str) -> dict:
        filter_dd = dict()
        for state, value_dd in dd.items():
            filter_dd.setdefault(state, dict())
            update_dd = {k: v[filter_key] for k, v in value_dd.items()}
            filter_dd[state].update(update_dd)
        return filter_dd

    def _convert_dict_to_frame(self, dd: dict, key: str, filter_key: Optional[str] = None) -> pd.DataFrame:
        if filter_key is not None:
            dd = self._filter_dict(dd=dd, filter_key=filter_key)

        df = pd.DataFrame(dd)
        df = pd.concat([df], axis="columns", keys=[key])
        return df

    def optimize(self) -> dict:
        self._set_value_dict()
        max_iterations = self.get_param(param="max_iterations")
        epsilon = self.get_param(param="epsilon")

        iteration = 1
        distance = np.Inf
        period_dd = dict()
        while True:
            if distance < epsilon:
                msg = f"convergence criteria met after {iteration} iterations"
                print(msg)
                break

            if iteration > max_iterations:
                msg = f"maximum number of iterations met"
                print(msg)
                break

            period_dd = self._calculate_period_value()
            prev_dd = self.get_param(param="value_next_dd")

            distance = self._calculate_distance(period_dd=period_dd, prev_dd=prev_dd)
            iteration += 1

            self._set_value_dict(value_next_dd=period_dd)

        return period_dd
