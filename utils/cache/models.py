#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Retention models."""

#-------------------------------------------------------------------------------


def gbt_regressor(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=seed)
