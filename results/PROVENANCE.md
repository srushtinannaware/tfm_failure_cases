# Result provenance

## Random context routing

`random_context_routing_results.csv` and its summary contain two groups of variants:

- anchor conditions at 200, 500, and 1,000 keys;
- bridge conditions from 150 to 1,150 keys in increments of 200.

These groups were generated during successive runs of the same task family while the sweep was being refined. They share the 3,000-row training budget, 400-row test budget, ten decoy features, floating-point key encoding, and 2% label-flip probability.

The current generator defines the bridge sweep. The anchor rows are retained because they were used during analysis, but they should not be described as output from one invocation of the current bridge loop.

## Feature grouping

`feature_role_switching_results.csv` contains:

- two role-sanity variants;
- six distance-sweep variants;
- five haystack-sweep variants.

The three groups answer different questions and should be plotted or summarized separately. In particular, the distance sweep changes feature position at fixed dimensionality, while the haystack sweep changes dimensionality with the active pair kept maximally separated.
