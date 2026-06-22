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
