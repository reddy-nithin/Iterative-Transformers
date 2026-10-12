           model  params (M)  GFLOPs  throughput (img/s)  final acc (%)  best acc (%)  epochs
        baseline       5.544    2.51                5653          86.77         86.83      10
       iterative       0.650    2.51                5773          34.76         34.76      10
   iterative_avg       0.650    2.51                5828          28.54         28.54      10
    iterative_b6       0.650    2.51                5820          30.70         30.70      10
iterative_lr5e-4       0.650    2.51                5565          46.02         46.02      10
         lora_r1       0.687    2.52                5348          27.66         27.66      10
         lora_r2       0.724    2.53                5352          27.51         27.57      10
     lora_r2zero       0.724    2.53                5546          28.74         28.74      10
         lora_r4       0.798    2.55                5565          27.22         27.22      10
         lora_r8       0.945    2.59                5334          30.69         30.69      10
         scratch       5.544    2.51                5789          28.87         28.87      10
