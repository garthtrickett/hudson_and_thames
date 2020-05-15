"""
Labeling techniques used in financial machine learning.
"""
import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + "/ProdigyAI/third_party_libraries/hudson_and_thames")

from mlfinlab.labeling.labeling import (add_vertical_barrier,
                                        apply_pt_sl_on_t1, barrier_touched,
                                        drop_labels, get_bins, get_first_touch_dates, get_events_from_first_touch_dates)
from mlfinlab.labeling.trend_scanning import trend_scanning_labels
from mlfinlab.labeling.tail_sets import TailSetLabels
