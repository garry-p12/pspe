| policy | burned % | burned % (pop-weighted) | reduction vs none % | reduction via fuel channel only % | treated % per day | over budget | real samples | forecast AUC-PR | final lambda |
|---|---|---|---|---|---|---|---|---|---|
| none | 10.83 | 14.76 | 0 | 0 | 0 | 0 | 8000 | 0.2802 |  |
| random | 10.33 | 14.06 | 4.659 | 0.6065 | 3 | 0 | 8000 | 0.2802 |  |
| greedy | 8.239 | 11.16 | 23.93 | 5.92 | 3 | 0 | 8000 | 0.2802 |  |
| pspe per-instance | 6.974 | 8.841 | 35.61 | 17.77 | 3 | 0 | 8000 | 0.2802 |  |
| pspe v2 | 9.07 | 12.02 | 16.26 | 3.434 | 2.938 | 0.4167 | 8000 | 0.2802 | 50 |
| unconstrained | 1.185 | 1.685 | 89.06 | 27.27 | 60.39 | 1 | 8000 | 0.2802 | 0 |

surrogate val AUC-PR 0.2802; budget 0.03
