"""The derivation shared by training and serving must mean the same thing twice.

campaign.data has two implementations of one rule: encode_pdays works on a
frame during training, contact_features works on one value during serving.
Two implementations of a rule is exactly the setup that produces train and
serve skew, so the rule is checked in both directions here.
"""
import math

import pandas as pd

from campaign.data import NEVER_CONTACTED, contact_features, encode_pdays


def test_never_contacted_is_a_gap_not_a_number():
    derived = contact_features(None)
    assert math.isnan(derived["pdays"])
    assert derived["previously_contacted"] == 0


def test_a_real_gap_is_kept():
    assert contact_features(6) == {"pdays": 6.0, "previously_contacted": 1}


def test_the_row_rule_and_the_frame_rule_agree(raw):
    sample = raw.head(500)
    frame_side = encode_pdays(sample)
    for (_, original), (_, derived) in zip(sample.iterrows(), frame_side.iterrows()):
        days = None if original["pdays"] == NEVER_CONTACTED else int(original["pdays"])
        row_side = contact_features(days)
        assert row_side["previously_contacted"] == derived["previously_contacted"]
        if math.isnan(row_side["pdays"]):
            assert pd.isna(derived["pdays"])
        else:
            assert row_side["pdays"] == derived["pdays"]
