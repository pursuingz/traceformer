# -*- coding: utf-8 -*-
"""汇总所有 n=14 held-out @45000 实验数据 -> CSV + Markdown 表。数字取自 test_result.md(+aug6win)。"""
import csv

# 列:每臂全部 n=14 指标。顺序:配置 -> 主精度 -> per-step(所有起点) -> per-step(start>0) -> 物理
# 值 None 表示该臂未记录该项。
ARMS = [
 # key, 名称, 架构, 参数M, 窗口, 课程/展开,
 # first, full, Chamfer, psA1,psA2,psA3,psA4, psZ1,psZ2,psZ3,
 # lossF, vMSE, aMSE, volRel%, volDrift%, floor%, floorDepth
 ("00","run23 (基线)","串行v1 8L","16.092","固定{0,5,10,15}","K=1 单步",
   4.356742e-4,4.981891e-3,7.382950e-2, 4.356741e-4,2.982977e-3,8.614889e-3,1.593847e-2,
   5.358088e-4,3.605935e-3,1.013112e-2, 2.903810e-2,1.472918e-4,2.350560e-5,6.300,8.698,0.649,1.973920e-3),
 ("01","课程+固定窗","串行v1 8L","16.092","固定{0,5,10,15}","课程K1→2→3+喂回",
   6.252360e-4,4.948577e-3,7.386877e-2, 6.252361e-4,3.368548e-3,8.594722e-3,1.553871e-2,
   7.736211e-4,4.061819e-3,1.014736e-2, 3.133843e-2,1.449982e-4,2.534771e-5,6.071,8.155,0.324,6.545817e-4),
 ("10","随机窗-only","串行v1 8L","16.092","随机[0,max]","K=1 单步",
   5.801420e-4,5.649160e-3,8.071447e-2, 5.801418e-4,3.233237e-3,8.811611e-3,1.811079e-2,
   7.138051e-4,3.795331e-3,9.776413e-3, 3.042274e-2,1.473273e-4,2.451168e-5,4.614,8.136,0.740,2.265623e-3),
 ("11","1d 课程+随机窗","串行v1 8L","16.092","随机[0,max]","课程K1→2→3+喂回",
   7.599083e-4,5.820629e-3,7.936898e-2, 7.599083e-4,4.085157e-3,1.015100e-2,1.865722e-2,
   9.535332e-4,5.080620e-3,1.212934e-2, 3.431213e-2,1.456918e-4,2.117075e-5,6.122,7.775,0.596,1.464179e-3),
 ("aug6win","aug6win 叠加增强","串行v1 8L","16.092","固定+2随机(叠加)","K=1 单步",
   3.984795e-4,4.457253e-3,7.327370e-2, 3.984794e-4,2.597374e-3,7.218780e-3,1.508026e-2,
   4.852840e-4,3.174982e-3,8.811774e-3, 2.670102e-2,1.238915e-4,2.033516e-5,6.665,10.499,0.579,1.252004e-3),
 ("v5","sft-tfs 双向并行","并行 4L","17.151","固定{0,5,10,15}","K=1 单步",
   5.755040e-4,6.032240e-3,7.884955e-2, 5.755039e-4,3.733807e-3,1.010652e-2,1.949949e-2,
   7.064337e-4,4.529455e-3,1.187655e-2, 2.987017e-2,1.614338e-4,2.643137e-5,7.458,9.391,0.771,None),
 ("v6","local-global 门控kNN","串行v1 8L","17.677","固定{0,5,10,15}+静态kNN k=16","K=1 单步",
   3.716034e-4,5.476996e-3,8.133792e-2, 3.716034e-4,2.739036e-3,8.205772e-3,1.714371e-2,
   4.373637e-4,3.056573e-3,8.448550e-3, 2.599940e-2,1.431966e-4,1.998253e-5,7.724,8.770,0.677,2.001381e-3),
 ("force0-s0","force0 seed=0","串行v1 8L","16.092","随机+必含start0","课程K1→2→3+喂回",
   8.518424e-4,6.633605e-3,8.114821e-2, 8.518425e-4,4.725724e-3,1.153986e-2,2.108048e-2,
   1.062984e-3,5.883230e-3,1.362130e-2, 3.579552e-2,1.586826e-4,2.243469e-5,7.713,9.519,0.581,1.679091e-3),
 ("force0-s1","force0 seed=1","串行v1 8L","16.092","随机+必含start0","课程K1→2→3+喂回",
   1.001065e-3,7.700335e-3,8.468367e-2, 1.001065e-3,5.636949e-3,1.366403e-2,2.373533e-2,
   1.238177e-3,6.895185e-3,1.597190e-2, 3.956149e-2,1.762395e-4,2.751417e-5,6.166,8.210,0.656,1.830859e-3),
]

COLS = ["arm","name","arch","params_M","window","curriculum",
        "first_chunk","full_rollout","Chamfer",
        "perstep_all_s1","perstep_all_s2","perstep_all_s3","perstep_all_s4",
        "perstep_nz_s1","perstep_nz_s2","perstep_nz_s3",
        "lossF_pred","vMSE","aMSE","vol_rel_pct","vol_drift_pct","floor_rate_pct","floor_depth"]

RUN23_FULL = 4.981891e-3


def fmt(v):
    if v is None: return "—"
    if isinstance(v, str): return v
    if abs(v) < 1: return f"{v:.3e}"
    return f"{v:.3f}"


# CSV (raw numbers)
with open("all_results_n14.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(COLS)
    for r in ARMS:
        w.writerow([r[i] if r[i] is not None else "" for i in range(len(COLS))])

# Markdown (grouped tables) -> all_results_n14.md
def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for row in rows:
        out += "| " + " | ".join(row) + " |\n"
    return out + "\n"


lines = ["# n=14 held-out 全实验数据(@checkpoint-45000,确定性 1-step)\n",
         "test=14 个同分布 held-out model;full-rollout 主指标(n=14),per-step start>0 n=42/28/14。\n\n"]

lines.append("## A. 训练配置\n")
lines.append(md_table(["臂","名称","架构","参数(M)","窗口","课程/展开"],
    [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in ARMS]))

lines.append("## B. 主精度(含 full-rollout vs run23 的 Δ)\n")
rowsB = []
for r in ARMS:
    d = "—" if r[0] == "00" else f"{(r[7]-RUN23_FULL)/RUN23_FULL*100:+.1f}%"
    rowsB.append([r[0], fmt(r[6]), fmt(r[7]), d, fmt(r[8])])
lines.append(md_table(["臂","first-chunk","full-rollout","Δvs run23","Chamfer"], rowsB))

lines.append("## C. per-step MSE(所有起点,n=56/42/28/14)\n")
lines.append(md_table(["臂","step1","step2","step3","step4"],
    [[r[0], fmt(r[9]), fmt(r[10]), fmt(r[11]), fmt(r[12])] for r in ARMS]))

lines.append("## D. per-step MSE(仅 start>0,公平起点,n=42/28/14)\n")
lines.append(md_table(["臂","step1","step2","step3"],
    [[r[0], fmt(r[13]), fmt(r[14]), fmt(r[15])] for r in ARMS]))

lines.append("## E. 物理合理性\n")
lines.append(md_table(["臂","loss_F(pred)","vMSE","aMSE","体积相对%","体积自漂移%","地面穿透%","穿透深度"],
    [[r[0], fmt(r[16]), fmt(r[17]), fmt(r[18]),
      f"{r[19]:.3f}", f"{r[20]:.3f}", f"{r[21]:.3f}", fmt(r[22])] for r in ARMS]))

with open("all_results_n14.md", "w", encoding="utf-8") as f:
    f.write("".join(lines))

print("saved all_results_n14.csv  &  all_results_n14.md")
print(f"arms={len(ARMS)}  cols={len(COLS)}")
