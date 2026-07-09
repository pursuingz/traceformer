"""按 E 分档拆解 eval 的 per-model CSV(eval.py per_model_csv=true 产出)。

用法:
  # 单表分档
  python analyze_E_bands.py vis_results_diffE_singleframe_geom_deform_d0001_per_model.csv
  # 双表按档对照(同一评测集,不同训练来源)
  python analyze_E_bands.py \
      vis_results_diffE_singleframe_geom_deform_d0001_per_model.csv  diffE-train \
      vis_results_fixedE_on_diffEtest_per_model.csv                  fixedE-train

分档(按 log10E,对齐 E 十进制 + 固定 E 训练点 1e6):
  soft  log10E < 5.0   (E < 1e5)
  mid   5.0 ~ 6.0      (1e5 ~ 1e6)
  stiff log10E >= 6.0  (E >= 1e6)
CSV 字段:model, log10E, nu, mse_full, vol_rel, drift_pred, drift_gt。
"""
import sys, csv, math

BANDS = [('soft', -1e9, 5.0), ('mid', 5.0, 6.0), ('stiff', 6.0, 1e9)]

def load(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d = {'model': r['model'], 'log10E': float(r['log10E']), 'nu': float(r.get('nu', 'nan'))}
            for k in ('mse_full', 'vol_rel', 'drift_pred', 'drift_gt', 'dp_f24', 'dg_f24'):
                d[k] = float(r[k]) if r.get(k) not in (None, '', 'nan') else float('nan')
            d['excess'] = d['drift_pred'] - d['drift_gt']
            # 末帧超额(f24=第25帧):逐帧列由 eval.py per_model_csv 落盘;老 CSV 无此列 → nan(列输出为 nan)
            d['excess_f24'] = d['dp_f24'] - d['dg_f24']
            rows.append(d)
    return rows

def band_of(lE):
    for name, lo, hi in BANDS:
        if lo <= lE < hi:
            return name
    return 'stiff'

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float('nan')

def summarize(rows, label):
    print(f"\n===== {label}  (n={len(rows)}) =====")
    print(f"  {'band':<6}{'n':>3}{'full-rollout':>15}{'vol_rel%':>10}{'drift_p%':>10}{'drift_gt%':>11}{'excess%':>10}{'exc_f24%':>10}")
    for name, lo, hi in BANDS:
        b = [r for r in rows if band_of(r['log10E']) == name]
        if not b:
            print(f"  {name:<6}{0:>3}{'—':>15}")
            continue
        print(f"  {name:<6}{len(b):>3}{mean([r['mse_full'] for r in b]):>15.3e}"
              f"{mean([r['vol_rel'] for r in b])*100:>10.2f}{mean([r['drift_pred'] for r in b])*100:>10.2f}"
              f"{mean([r['drift_gt'] for r in b])*100:>11.2f}{mean([r['excess'] for r in b])*100:>10.2f}"
              f"{mean([r['excess_f24'] for r in b])*100:>10.2f}")
    allrows = rows
    print(f"  {'ALL':<6}{len(allrows):>3}{mean([r['mse_full'] for r in allrows]):>15.3e}"
          f"{mean([r['vol_rel'] for r in allrows])*100:>10.2f}{mean([r['drift_pred'] for r in allrows])*100:>10.2f}"
          f"{mean([r['drift_gt'] for r in allrows])*100:>11.2f}{mean([r['excess'] for r in allrows])*100:>10.2f}"
          f"{mean([r['excess_f24'] for r in allrows])*100:>10.2f}")

def compare(rowsA, labA, rowsB, labB):
    """双表:同评测集不同训练来源,按档比 full-rollout。"""
    print(f"\n===== 按档对照 full-rollout:{labA} vs {labB} =====")
    print(f"  {'band':<6}{'n':>3}{labA:>16}{labB:>16}{'B/A':>8}")
    idxB = {r['model']: r for r in rowsB}
    for name, lo, hi in BANDS:
        a = [r for r in rowsA if band_of(r['log10E']) == name]
        if not a:
            continue
        b = [idxB[r['model']] for r in a if r['model'] in idxB]
        mA, mB = mean([r['mse_full'] for r in a]), mean([r['mse_full'] for r in b]) if b else float('nan')
        ratio = mB / mA if mA and not math.isnan(mA) and mA > 0 else float('nan')
        print(f"  {name:<6}{len(a):>3}{mA:>16.3e}{mB:>16.3e}{ratio:>8.2f}")
    mA, mB = mean([r['mse_full'] for r in rowsA]), mean([r['mse_full'] for r in rowsB])
    print(f"  {'ALL':<6}{len(rowsA):>3}{mA:>16.3e}{mB:>16.3e}{(mB/mA if mA>0 else float('nan')):>8.2f}")

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    csv1 = args[0]; lab1 = args[1] if len(args) > 1 and not args[1].endswith('.csv') else 'run1'
    rows1 = load(csv1); summarize(rows1, lab1)
    rest = args[2:] if (len(args) > 1 and not args[1].endswith('.csv')) else args[1:]
    if rest:
        csv2 = rest[0]; lab2 = rest[1] if len(rest) > 1 else 'run2'
        rows2 = load(csv2); summarize(rows2, lab2)
        compare(rows1, lab1, rows2, lab2)
