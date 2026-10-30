  run                                          layout  params (M)  epochs  final top-1  best top-1  sep last block (test)  sep last body loop (test)
T11-a           T11-a: head b0, body b0 x10, tail b11       1.540    30.0        62.05       62.05                   1.80                       1.79
T11-b           T11-b: head b0, body b6 x10, tail b11       1.540    30.0        63.47       63.62                   2.04                       2.28
T11-c           T11-c: head b0, body b1 x10, tail b11       1.540    30.0        68.59       68.63                   1.38                       1.61
T11-d  T11-d: unrolling n=3 (b0, b1, b2 x10), no tail       1.540    30.0        74.11       74.13                   1.15                       1.15
  ref baseline, 12 distinct blocks, fine-tuned (T5-a)       5.544     NaN        86.77         NaN                    NaN                        NaN
  ref           iter b0, block 0 x12, 30 ep (T4/T5-b)       0.650     NaN        58.06         NaN                    NaN                        NaN
  ref                     per-loop LoRA r8 on b0 (T4)       0.945     NaN        61.33         NaN                    NaN                        NaN
  ref      head + body x10 + tail from scratch (T9-a)       1.540     NaN        44.87         NaN                    NaN                        NaN
  ref              unrolling n=3, lr 5e-4, 10 ep (T3)       1.540     NaN        67.12         NaN                    NaN                        NaN
