"""
MlFinlab helps portfolio managers and traders who want to leverage the power of machine learning by providing
reproducible, interpretable, and easy to use tools.

Adding MlFinLab to your companies pipeline is like adding a department of PhD researchers to your team.
"""
import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + "/ProdigyAI/third_party_libraries/hudson_and_thames")

import mlfinlab.cross_validation as cross_validation
import mlfinlab.data_structures as data_structures
import mlfinlab.datasets as datasets
import mlfinlab.multi_product as multi_product
from mlfinlab.filters import filters
import mlfinlab.labeling as labeling
import mlfinlab.features.fracdiff as fracdiff
import mlfinlab.sample_weights as sample_weights
import mlfinlab.sampling as sampling
import mlfinlab.bet_sizing as bet_sizing
import mlfinlab.util as util
import mlfinlab.structural_breaks as structural_breaks
import mlfinlab.feature_importance as feature_importance
import mlfinlab.ensemble as ensemble
import mlfinlab.portfolio_optimization as portfolio_optimization
import mlfinlab.clustering as clustering
import mlfinlab.microstructural_features as microstructural_features
from mlfinlab.backtest_statistics import backtests
from mlfinlab.backtest_statistics import statistics
