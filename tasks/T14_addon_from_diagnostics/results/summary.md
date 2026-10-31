| Run | Model | Params | Learned η | Final top-1 | Best top-1 | Loop-5 acc | Flip 9→10 (%) | Fewest loops within 1 pt of loop 10 | Mean rel_cls loops 1-5 | Late-loop cos(A_t, A_t+1) | Mean H(A_t) | Sep. last loop (test) | Min loop sep. (test, call) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T13-d | base: every loop + margin | 1,539,940 | – | 73.17 | 73.17 | 73.30 | 1.58 | kl 0.0316: 4.07 loops (73.13%) vs fixed 4 | 0.512 | 0.995 | 4.76 | 1.40 | 1.38 (9) |
| T14-a | base + coverage mask | 1,539,941 | 0.6375 | 72.89 | 72.89 | 73.20 | 1.66 | kl 0.0316: 4.08 loops (72.83%) vs fixed 4 | 0.517 | 0.995 | 4.75 | 1.38 | 1.34 (8) |
