# Results overview

## Wide, short regression

The targeted benchmark holds the sample budget at 100 training rows and 500 test rows while sweeping 100, 200, and 500 features. Approximately 10% of features contribute to a noisy linear regression target.

The strongest comparative separation occurs at 200 features. TabPFN-2.5 wins all three seeds and reaches mean R² 0.956. The result demonstrates a reproducible failure regime for the evaluated alternatives under this generator, not a universal model ranking.

## Multi-condition XOR

The XOR benchmark requires learning a logical interaction rather than a marginal feature effect. Adding irrelevant features dilutes the interaction signal. CatBoost remains stronger than the evaluated TFMs in the highlighted noisy condition, showing that the task has not simply become impossible.

## Random context routing

Each categorical key maps to a binary label. Increasing the number of keys reduces the expected number of training examples per key. TabICL v2 retains substantially higher classification quality in the sparse-key regime.

## Interpretation

The three benchmarks expose different inductive biases:

- distributed signal in a short-wide regression table;
- logical interaction under irrelevant-feature dilution;
- sparse categorical lookup from in-context examples.

No single model dominates all three. The appropriate conclusion is conditional robustness, not an overall winner.
