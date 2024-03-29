import json

import pytest
from visitor import Visitor

import librun
import libstop
from libdatasets import *
from libstop import *
from libadversarial import uncertainty_stop
from libstop import rank_stop_conds


@pytest.mark.results
def test_eval(results, dataset_dir, temp_out, snapshot):
    results_plots, classifiers = results

    conditions = {
        "GOAL": partial(ZPS, order=2),
        "SSNCut": SSNCut,
        "SC_entropy_mcs": SC_entropy_mcs,
        "SC_oracle_acc": SC_oracle_acc_mcs,
        "SC_mes": SC_mes,
        "Stabilizing Predictions": stabilizing_predictions,
        "Performance Convergence": performance_convergence,
        "Uncertainty Convergence": uncertainty_convergence,
        "Max Confidence": max_confidence,
        "EVM": EVM,
        "VM": VM,
        "Contradictory Information": contradictory_information,
        "Classification Change": classification_change,
        "Overall Uncertainty": overall_uncertainty,
    }

    stop_conditions, stop_results = libstop.eval_stopping_conditions(
        results_plots, classifiers, conditions=conditions
    )

    snapshot.assert_match(json.dumps(Rounder(3).visit(stop_results)), "test_eval.json")


class Rounder(Visitor):
    """
    Walk an object, rounding all floating point values to some number of decimal places.
    """

    def __init__(self, dp):
        self.dp = dp

    def visit_float(self, f):
        return round(f, self.dp)

    def visit_int(self, i):
        return i

    def visit_str(self, s):
        return s

    def visit_NoneType(self, n):
        return n

    def visit_tuple(self, l):
        return tuple(self.visit(x) for x in l)

    def visit_list(self, l):
        return [self.visit(x) for x in l]

    def visit_dict(self, d):
        return {self.visit(k): self.visit(v) for k, v in d.items()}
