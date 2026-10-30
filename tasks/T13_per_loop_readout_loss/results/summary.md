| Run | Layout | Loss | Params (M) | Final top-1 | Best top-1 | Loop-5 acc | Flip 9→10 (%) | Fewest loops within 1 pt of loop 10 | Mean rel_cls loops 1-5 | Sep. last body loop (test) | Min body sep. (test, call) | Sep. last block (test) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T11-c | T11-c | ref: final loss only | 1.540 | 68.59 | 68.63 | 24.54 | 9.05 | confidence 0.7: 8.78 loops (67.99%) vs fixed 9 | 0.297 | 1.61 | 1.61 (10) | 1.38 |
| T13-a | T11-c | readout every loop | 1.540 | 68.85 | 68.92 | 69.34 | 2.54 | patience 2: 3.54 loops (68.64%) vs fixed 4 | 0.328 | 1.89 | 1.84 (7) | 1.29 |
| T13-b | T11-c | every loop + margin | 1.540 | 68.85 | 68.95 | 67.56 | 2.71 | kl 0.01: 4.99 loops (68.15%) vs fixed 6 | 0.250 | 2.14 | 2.00 (6) | 1.54 |
| T11-d | T11-d | ref: final loss only | 1.540 | 74.11 | 74.13 | 49.67 | 5.23 | confidence 0.5: 7.55 loops (73.47%) vs fixed 8 | 0.299 | 1.15 | 1.06 (9) | 1.15 |
| T13-c | T11-d | readout every loop | 1.540 | 72.82 | 72.84 | 73.27 | 1.49 | kl 0.1: 3.44 loops (72.63%) vs fixed 4 | 0.595 | 1.27 | 1.23 (8) | 1.27 |
| T13-d | T11-d | every loop + margin | 1.540 | 73.17 | 73.17 | 73.30 | 1.58 | kl 0.0316: 4.07 loops (73.13%) vs fixed 4 | 0.512 | 1.40 | 1.38 (9) | 1.40 |
