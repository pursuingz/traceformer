eval_23.yaml结果：
===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 4.356742e-04   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 4.981891e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 7.382950e-02
  MSE per step    : step1=1.352701e-04, step2=1.737061e-03, step3=7.098657e-03, step4=1.593847e-02
  per-step(所有起点): step1=4.356741e-04(n=56), step2=2.982977e-03(n=42), step3=8.614889e-03(n=28), step4=1.593847e-02(n=14)
  per-step(仅start>0): step1=5.358088e-04(n=42), step2=3.605935e-03(n=28), step3=1.013112e-02(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 2.903810e-02
  loss_F (gt ref) : 8.513588e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.472918e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.350560e-05   (二阶差, vs GT)
  体积相对误差    : 6.300%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 8.698%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.649%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 1.973920e-03   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)



eval_v1_randwin_only.yaml结果：
===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 5.801420e-04   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 5.649160e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 8.071447e-02
  MSE per step    : step1=1.791522e-04, step2=2.109051e-03, step3=7.846812e-03, step4=1.811079e-02
  per-step(所有起点): step1=5.801418e-04(n=56), step2=3.233237e-03(n=42), step3=8.811611e-03(n=28), step4=1.811079e-02(n=14)
  per-step(仅start>0): step1=7.138051e-04(n=42), step2=3.795331e-03(n=28), step3=9.776413e-03(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 3.042274e-02
  loss_F (gt ref) : 8.513581e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.473273e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.451168e-05   (二阶差, vs GT)
  体积相对误差    : 4.614%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 8.136%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.740%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 2.265623e-03   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)



eval_v1_curriculum_fixedwin.yaml结果：
===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 6.252360e-04   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 4.948577e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 7.386877e-02
  MSE per step    : step1=1.800809e-04, step2=1.982005e-03, step3=7.042084e-03, step4=1.553871e-02
  per-step(所有起点): step1=6.252361e-04(n=56), step2=3.368548e-03(n=42), step3=8.594722e-03(n=28), step4=1.553871e-02(n=14)
  per-step(仅start>0): step1=7.736211e-04(n=42), step2=4.061819e-03(n=28), step3=1.014736e-02(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 3.133843e-02
  loss_F (gt ref) : 8.513581e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.449982e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.534771e-05   (二阶差, vs GT)
  体积相对误差    : 6.071%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 8.155%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.324%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 6.545817e-04   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)



eval_v1_rollout_curriculum.yaml结果：
===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 7.599083e-04   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 5.820629e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 7.936898e-02
  MSE per step    : step1=1.790336e-04, step2=2.094229e-03, step3=8.172668e-03, step4=1.865722e-02
  per-step(所有起点): step1=7.599083e-04(n=56), step2=4.085157e-03(n=42), step3=1.015100e-02(n=28), step4=1.865722e-02(n=14)
  per-step(仅start>0): step1=9.535332e-04(n=42), step2=5.080620e-03(n=28), step3=1.212934e-02(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 3.431213e-02
  loss_F (gt ref) : 8.513581e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.456918e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.117075e-05   (二阶差, vs GT)
  体积相对误差    : 6.122%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 7.775%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.596%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 1.464179e-03   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)



eval_v1_curriculum_randwin_force0.yaml结果
===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 8.518424e-04   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 6.633605e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 8.114821e-02
  MSE per step    : step1=2.184173e-04, step2=2.410715e-03, step3=9.458420e-03, step4=2.108048e-02
  per-step(所有起点): step1=8.518425e-04(n=56), step2=4.725724e-03(n=42), step3=1.153986e-02(n=28), step4=2.108048e-02(n=14)
  per-step(仅start>0): step1=1.062984e-03(n=42), step2=5.883230e-03(n=28), step3=1.362130e-02(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 3.579552e-02
  loss_F (gt ref) : 8.513588e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.586826e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.243469e-05   (二阶差, vs GT)
  体积相对误差    : 7.713%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 9.519%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.581%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 1.679091e-03   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)



eval_v1_curriculum_randwin_force0_seed1.yaml结果

===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 1.001065e-03   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 7.700335e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 8.468367e-02
  MSE per step    : step1=2.897275e-04, step2=3.120477e-03, step3=1.135615e-02, step4=2.373533e-02
  per-step(所有起点): step1=1.001065e-03(n=56), step2=5.636949e-03(n=42), step3=1.366403e-02(n=28), step4=2.373533e-02(n=14)
  per-step(仅start>0): step1=1.238177e-03(n=42), step2=6.895185e-03(n=28), step3=1.597190e-02(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 3.956149e-02
  loss_F (gt ref) : 8.513588e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.762395e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.751417e-05   (二阶差, vs GT)
  体积相对误差    : 6.166%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 8.210%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.656%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 1.830859e-03   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)


========================================
eval_v5_4L.yaml结果 (@45000, data_test n=14) —— Phase 4 v5 双向并行,证伪
===== eval metrics =====
  windows: 56 total, 14 full-horizon
  MSE first-chunk : 5.755040e-04
  MSE full-rollout: 6.032240e-03
  Chamfer  (mean) : 7.884955e-02
  MSE per step    : step1=1.827145e-04, step2=2.142511e-03, step3=8.336484e-03, step4=1.949949e-02
  per-step(所有起点): step1=5.755039e-04(n=56), step2=3.733807e-03(n=42), step3=1.010652e-02(n=28), step4=1.949949e-02(n=14)
  per-step(仅start>0): step1=7.064337e-04(n=42), step2=4.529455e-03(n=28), step3=1.187655e-02(n=14), step4=0(n=0)
  loss_F (pred)   : 2.987017e-02 / (gt ref) 8.513581e-03
  vMSE 1.614338e-04 / aMSE 2.643137e-05
  体积相对 7.458% / 体积自漂移 9.391% / 地面穿透 0.771%



========================================
eval_v6_local_global_8L.yaml结果 (@45000, data_test n=14) —— v1串行 + 门控静态kNN局部消息 + aug6win,混合(start>0赢/start=0败)
===== eval metrics =====
  windows: 56 total, 14 full-horizon
  MSE first-chunk : 3.716e-04
  MSE full-rollout: 5.477e-03
  Chamfer  (mean) : 8.134e-02
  per-step(所有起点): step1=3.716e-04, step2=2.739e-03, step3=8.206e-03, step4=1.714e-02
  per-step(仅start>0): step1=4.374e-04(n=42), step2=3.057e-03(n=28), step3=8.449e-03(n=14)
  loss_F (pred)   : 2.600e-02
  vMSE 1.432e-04 / aMSE 1.998e-05
  体积相对 7.724% / 体积自漂移 8.770% / 地面穿透 0.677% / 穿透深度 2.001e-03


========================================
eval_v6b_strain_gated_8L.yaml结果 (@45000, data_test n=14) —— v6 + strain-gate(边长变化率门控局部消息),证伪·全面更差
===== eval metrics =====
  windows: 56 total, 14 full-horizon (rollout 指标分母)
  MSE first-chunk : 4.389255e-04   (全 56 窗口, 逐样本对齐)
  MSE full-rollout: 6.325590e-03   (仅 14 全程窗口)
  Chamfer  (mean) : 8.401290e-02
  MSE per step    : step1=1.792536e-04, step2=2.217739e-03, step3=8.835914e-03, step4=2.039505e-02
  per-step(所有起点): step1=4.389254e-04(n=56), step2=3.379530e-03(n=42), step3=1.016612e-02(n=28), step4=2.039505e-02(n=14)
  per-step(仅start>0): step1=5.254827e-04(n=42), step2=3.960426e-03(n=28), step3=1.149633e-02(n=14), step4=0.000000e+00(n=0)
  loss_F (pred)   : 2.636895e-02
  loss_F (gt ref) : 8.513588e-03
  --- 物理合理性 ---
  速度误差 vMSE   : 1.534680e-04   (帧差, vs GT, 14 全程窗口)
  加速度误差 aMSE : 2.180011e-05   (二阶差, vs GT)
  体积相对误差    : 7.799%   (|Vp-Vg|/Vg, 350 帧, 凸包)
  体积自漂移      : 9.120%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
  地面穿透率      : 0.889%   (预测帧占全部点比例, 全 56 窗口)
  地面穿透深度    : 3.226173e-03   (归一化单位)
  地面穿透率(GT)  : 0.000%   (参考, 应≈0)

---

## velocity_v1_8L — n=14 重测(@45000,data_test,2026-06-23,commit 0a25f18)
旧 @55000/n=4 口径作废,本块为 n=14 同表口径。

```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 4.963422e-04   (全 56 窗口, 逐样本对齐)
MSE full-rollout: 6.510323e-03   (仅 14 全程窗口)
Chamfer  (mean) : 7.778424e-02
MSE per step    : step1=1.534455e-04, step2=2.066155e-03, step3=8.819002e-03, step4=2.151302e-02
per-step(所有起点): step1=4.963422e-04(n=56), step2=3.270838e-03(n=42), step3=1.007081e-02(n=28), step4=2.151302e-02(n=14)
per-step(仅start>0): step1=6.106411e-04(n=42), step2=3.873179e-03(n=28), step3=1.132262e-02(n=14), step4=0.000000e+00(n=0)
loss_F (pred)   : 2.833352e-02
loss_F (gt ref) : 8.513581e-03
--- 物理合理性 ---
速度误差 vMSE   : 1.756721e-04   (帧差, vs GT, 14 全程窗口)
加速度误差 aMSE : 1.873037e-05   (二阶差, vs GT)
体积相对误差    : 6.090%   (|Vp-Vg|/Vg, 350 帧, 凸包)
体积自漂移      : 9.057%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
地面穿透率      : 1.045%   (预测帧占全部点比例, 全 56 窗口)
地面穿透深度    : 5.850166e-03   (归一化单位)
地面穿透率(GT)  : 0.000%   (参考, 应≈0)
```

## v8_physics_slice_8L — n=14(@45000,data_test,2026-06-24,代码 commit 22b81df / 台账 18df5de)
基线=aug6win(同用 train_extra_random_windows=2)。Transolver slice 瓶颈替换点级空间注意力,16.158M。

```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 5.907095e-04   (全 56 窗口, 逐样本对齐)
MSE full-rollout: 5.685990e-03   (仅 14 全程窗口)
Chamfer  (mean) : 7.748289e-02
MSE per step    : step1=1.349542e-04, step2=1.937056e-03, step3=8.217814e-03, step4=1.814013e-02
per-step(所有起点): step1=5.907095e-04(n=56), step2=3.990817e-03(n=42), step3=1.060659e-02(n=28), step4=1.814013e-02(n=14)
per-step(仅start>0): step1=7.426280e-04(n=42), step2=5.017695e-03(n=28), step3=1.299537e-02(n=14), step4=0.000000e+00(n=0)
loss_F (pred)   : 3.322764e-02
loss_F (gt ref) : 8.513588e-03
--- 物理合理性 ---
速度误差 vMSE   : 1.427770e-04   (帧差, vs GT, 14 全程窗口)
加速度误差 aMSE : 2.134926e-05   (二阶差, vs GT)
体积相对误差    : 5.507%   (|Vp-Vg|/Vg, 350 帧, 凸包)
体积自漂移      : 7.465%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
地面穿透率      : 0.732%   (预测帧占全部点比例, 全 56 窗口)
地面穿透深度    : 1.738645e-03   (归一化单位)
```

## v9_dual_graph_8L — n=14(@45000,data_test,2026-06-25,代码 commit 9f0edd2 / 台账 9f0edd2)
基线=aug6win(同用 train_extra_random_windows=2)+ v6(等参 17.677M,唯一差 rest 静止图)。rest 帧 kNN ∥ current 帧 kNN 双图局部残差,共享 MLP zero-gate。

```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 3.953153e-04   (全 56 窗口, 逐样本对齐)
MSE full-rollout: 5.040304e-03   (仅 14 全程窗口)
Chamfer  (mean) : 7.804323e-02
MSE per step    : step1=1.568224e-04, step2=1.926997e-03, step3=7.371496e-03, step4=1.574621e-02
per-step(所有起点): step1=3.953153e-04(n=56), step2=2.873836e-03(n=42), step3=8.160165e-03(n=28), step4=1.574621e-02(n=14)
per-step(仅start>0): step1=4.748130e-04(n=42), step2=3.347255e-03(n=28), step3=8.948832e-03(n=14), step4=0.000000e+00(n=0)
loss_F (pred)   : 2.686850e-02
loss_F (gt ref) : 8.513581e-03
--- 物理合理性 ---
速度误差 vMSE   : 1.314446e-04   (帧差, vs GT, 14 全程窗口)
加速度误差 aMSE : 1.990551e-05   (二阶差, vs GT)
体积相对误差    : 7.324%   (|Vp-Vg|/Vg, 350 帧, 凸包)
体积自漂移      : 9.272%   (|Vp(t)-Vp(0)|/Vp(0), 近不可压应小)
地面穿透率      : 0.737%   (预测帧占全部点比例, 全 56 窗口)
地面穿透深度    : 2.169066e-03   (归一化单位)
地面穿透率(GT)  : 0.000%   (参考, 应≈0)
```

## singleframe / run23 / randwin-only — 同口径三臂(@45000,data_test,n=14,2026-06-27,代码 commit 6d7bc04)
> 三臂均在 `6d7bc04`(eval 噪声走 generator,可复现)后代码重 eval。singleframe = output_frames=1 单帧自回归 + 随机窗口(commit ae5ba07/dd21494/a5d73cc)。first-chunk/per-step 在 out=1 不可比;主判据 = full-rollout + per-abs-frame。

### singleframe (output=1 + 随机窗口)
```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 1.160693e-05   (全 56 窗口, 逐样本对齐; out=1 1帧桶不可比)
MSE full-rollout: 4.532404e-03   (仅 14 全程窗口)
Chamfer  (mean) : 6.265244e-02
MSE per abs-frame: f5=2.986225e-06(n=14), f10=4.775387e-04(n=14), f15=4.019383e-03(n=14), f20=1.065516e-02(n=14)
per-step(所有起点): step1..4=0(对齐 off-by-4), step5=1.160693e-05(=first-chunk) ... step20=1.065516e-02
loss_F (pred)   : 0.000000e+00   (out=1 DeformLoss 需 >=3 帧, N/A)
loss_F (gt ref) : 0.000000e+00
--- 物理合理性 ---
速度误差 vMSE   : 1.237089e-04
加速度误差 aMSE : 8.872184e-06
体积相对误差    : 10.457%
体积自漂移      : 14.742%
地面穿透率      : 0.911%
地面穿透深度    : 4.046942e-03
地面穿透率(GT)  : 0.000%
```

### sfG = singleframe + 几何正则 (laplacian0.5/edge1.0, commit 12bda54, @45000, data_test, n=14, 2026-06-29)
```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 1.016152e-05   (out=1 1帧桶不可比)
MSE full-rollout: 3.503756e-03   (仅 14 全程窗口; −22.7% vs sf 4.532e-3 / −30.2% vs run23 5.019e-3)
Chamfer  (mean) : 5.861589e-02
MSE per abs-frame: f5=3.656353e-06(n=14), f10=3.148597e-04(n=14), f15=2.636222e-03(n=14), f20=8.291370e-03(n=14)
loss_F (pred)   : 0.000000e+00   (out=1, N/A)
loss_F (gt ref) : 0.000000e+00
--- 物理合理性 ---
速度误差 vMSE   : 9.290832e-05
加速度误差 aMSE : 6.736040e-06
体积相对误差    : 9.550%
体积自漂移      : 12.714%   (sf 14.742% → −13.8%, 仍 +45.9% vs run23 8.7%)
地面穿透率      : 0.944%
地面穿透深度    : 5.505066e-03   (sf 4.047e-3 → +36%, 变差)
地面穿透率(GT)  : 0.000%
```
> sfG vs sf 单变量(只动 2 lambda):MSE/运动/Chamfer/体积全降,唯穿透深度升。per-abs 晚帧赢最多(f20 −22% vs sf)。**渲染待判**——MSE 非充分质量代理(同 sf 教训)。

### 体积自漂移 GT 基线 + 超额(commit 37f93c8, 2026-06-29)
> GT 自漂移与模型无关,三臂通用 = **5.014%**(凸包体积非物质体积,弹性体 nu=0.4 大形变下 GT 本就漂)。超额 = pred − GT = 模型额外引入的体积不稳定。
```
体积自漂移(GT)  : 5.014%
                 run23  pred 8.698% → 超额 +3.684%
                 sf     pred 14.742% → 超额 +9.728%  (run23 的 2.64×)
                 sfG    pred 12.714% → 超额 +7.700%  (run23 的 2.09×; 超额口径 vs sf −20.9%)
```
> 结论:① GT 基线≠0,绝对值口径高估病态(方法学质疑成立);② 三臂超额均 >0 →「run23 形变不足」反转证伪、「run23 体积最稳」成立;③「sf 体积保真差」成立不撤回;④ sfG 几何正则疗效超额口径 −20.9% > 绝对值 −13.8%。

### run23 (同口径重 eval)
```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 4.363965e-04
MSE full-rollout: 5.019076e-03
Chamfer  (mean) : 7.401446e-02
MSE per step    : step1=1.357289e-04, step2=1.743680e-03, step3=7.171936e-03, step4=1.604404e-02
MSE per abs-frame: f5=1.142846e-05(n=14), f10=6.341561e-04(n=14), f15=4.250654e-03(n=14), f20=1.182760e-02(n=14)
per-step(所有起点): step1=4.363966e-04(n=56), step2=2.978254e-03(n=42), step3=8.596060e-03(n=28), step4=1.604404e-02(n=14)
per-step(仅start>0): step1=5.366190e-04(n=42), step2=3.595541e-03(n=28), step3=1.002018e-02(n=14)
loss_F (pred)   : 2.901559e-02
loss_F (gt ref) : 8.513581e-03
--- 物理合理性 ---
速度误差 vMSE   : 1.478361e-04
加速度误差 aMSE : 2.403502e-05
体积相对误差    : 6.326%
体积自漂移      : 8.716%
地面穿透率      : 0.645%
地面穿透深度    : 1.964012e-03
地面穿透率(GT)  : 0.000%
```

### randwin-only (10) (同口径重 eval)
```
===== eval metrics =====
windows: 56 total, 14 full-horizon (rollout 指标分母)
MSE first-chunk : 5.791268e-04
MSE full-rollout: 5.677542e-03
Chamfer  (mean) : 8.091000e-02
MSE per step    : step1=1.786927e-04, step2=2.115013e-03, step3=7.894638e-03, step4=1.819937e-02
MSE per abs-frame: f5=2.039766e-05(n=14), f10=7.878247e-04(n=14), f15=4.988688e-03(n=14), f20=1.301424e-02(n=14)
per-step(所有起点): step1=5.791268e-04(n=56), step2=3.232925e-03(n=42), step3=8.840105e-03(n=28), step4=1.819937e-02(n=14)
per-step(仅start>0): step1=7.126047e-04(n=42), step2=3.791881e-03(n=28), step3=9.785569e-03(n=14)
loss_F (pred)   : 3.041286e-02
loss_F (gt ref) : 8.513588e-03
--- 物理合理性 ---
速度误差 vMSE   : 1.477021e-04
加速度误差 aMSE : 2.458166e-05
体积相对误差    : 4.640%
体积自漂移      : 8.180%
地面穿透率      : 0.742%
地面穿透深度    : 2.267795e-03
地面穿透率(GT)  : 0.000%
```
