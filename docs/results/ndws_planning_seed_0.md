| policy | burned % | burned % (pop-weighted) | reduction vs none % | reduction via fuel channel only % | treated % per day | over budget | real samples | forecast AUC-PR | final lambda |
|---|---|---|---|---|---|---|---|---|---|
| none | 10.62 | 14.43 | 0 | 0 | 0 | 0 | 8000 | 0.2643 |  |
| random | 10.17 | 13.83 | 4.226 | 0.2648 | 3 | 0 | 8000 | 0.2643 |  |
| greedy | 7.649 | 10.46 | 27.96 | 8.528 | 3 | 0 | 8000 | 0.2643 |  |
| pspe per-instance | 6.628 | 8.508 | 37.57 | 19.34 | 3 | 0 | 8000 | 0.2643 |  |
| pspe v2 | 9.828 | 13.3 | 7.439 | 1.887 | 0.9966 | 0.0947 | 8000 | 0.2643 | 50 |
| unconstrained | 1.324 | 1.845 | 87.53 | 18.04 | 70.1 | 0.9993 | 8000 | 0.2643 | 0 |

surrogate val AUC-PR 0.2643; budget 0.03
