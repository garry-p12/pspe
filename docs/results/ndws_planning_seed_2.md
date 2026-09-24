| policy | burned % | burned % (pop-weighted) | reduction vs none % | reduction via fuel channel only % | treated % per day | over budget | real samples | forecast AUC-PR | final lambda |
|---|---|---|---|---|---|---|---|---|---|
| none | 10.81 | 14.81 | 0 | 0 | 0 | 0 | 8000 | 0.2756 |  |
| random | 10.3 | 14.09 | 4.723 | 1.045 | 3 | 0 | 8000 | 0.2756 |  |
| greedy | 8.605 | 11.55 | 20.39 | 2.762 | 3 | 0 | 8000 | 0.2756 |  |
| pspe per-instance | 6.881 | 8.591 | 36.33 | 20.91 | 2.999 | 0 | 8000 | 0.2756 |  |
| pspe v2 | 9.422 | 12.55 | 12.83 | 2.281 | 2.728 | 0.0113 | 8000 | 0.2756 | 50 |
| unconstrained | 2.092 | 2.755 | 80.64 | 27.88 | 54.6 | 1 | 8000 | 0.2756 | 0 |

surrogate val AUC-PR 0.2756; budget 0.03
