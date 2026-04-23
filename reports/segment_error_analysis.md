# Segment Error Analysis

Lower recall means more observed defaults are missed inside that borrower group.

## Weakest Recall Segments
- late_payment_bucket=none: recall=0.014, FN=632, default_rate=0.027
- age_bucket=66+: recall=0.291, FN=90, default_rate=0.022
- income_bucket=income_q4: recall=0.363, FN=219, default_rate=0.046

## Highest Observed Risk Segments
- late_payment_bucket=3+: default_rate=0.446, avg_probability=0.443
- late_payment_bucket=1-2: default_rate=0.155, avg_probability=0.151
- age_bucket=18-35: default_rate=0.121, avg_probability=0.109