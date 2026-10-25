"""Summarize provenance and validation without re-running physics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('directory')
args=parser.parse_args()
root=Path(args.directory).resolve()
r=json.loads((root/'report.json').read_text())
a=r['attempts'];good=[x for x in a if x['valid']]
if not good:raise SystemExit('No accepted episodes.')
steps=np.array([x['steps'] for x in good])
coverage=np.array([x['final_coverage'] for x in good])
fields=['letter','seed','valid','final_coverage','steps','duration_seconds','wall_seconds','max_state_error','max_contact_penetration_px','path']
with (root/'episodes.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(a)
size=sum(p.stat().st_size for p in (root/'episodes').glob('*.npz'))
lines=[
'# Push Any Letter：自动规划示范数据', '',
'来源：`model_predictive_planner`。控制器使用物体精确状态和已知仿真模型；不是人工遥操作数据，也不是训练后的策略输出。', '',
f'- 字母覆盖：{sum(n>0 for n in r["per_letter"].values())}/{len(r["per_letter"])}；配额全部完成：{r["complete"]}。',
f'- 通过验证：{len(good)}/{len(a)} 次尝试。每字母目标数量：{r["config"]["episodes"]}。',
f'- 最终覆盖率：最小 {100*coverage.min():.2f}%，平均 {100*coverage.mean():.2f}%，最大 {100*coverage.max():.2f}%。',
f'- 总控制步：{steps.sum():,}；20 Hz 下累计仿真时间 {steps.sum()/20:.1f} 秒。',
f'- 单条长度：{steps.min()}–{steps.max()} 步，中位数 {np.median(steps):.0f} 步。',
f'- 完整状态回放最大误差：{max(x["max_state_error"] for x in good):.6g}。',
f'- 采样控制时刻最大接触穿透：{max(x["max_contact_penetration_px"] for x in good):.3f} 像素（验收上限 3 像素）。',
f'- NPZ 数据体积：{size/1024/1024:.2f} MiB。',
f'- 所有尝试合计计算耗时：{sum(x["wall_seconds"] for x in a):.1f} 秒。', '',
'逐条结果见 `episodes.csv`；完整配置、失败尝试和验证指标见 `report.json`。字母名称使用 Unicode 编码写入文件名，避免大小写冲突。', '',
'每个 `.npz` 包含 `observations`、`actions`、`next_observations`、`rewards`、`terminated`、`truncated` 和 JSON `metadata`。启用图像后另有动作前 RGB `images` 及秒数 `timestamps`。使用 `np.load(path, allow_pickle=False)` 读取。', '',
'独立验证会从 seed 重建环境，只回放保存的动作，逐步比较状态、奖励、结束标记、动作边界、场地边界与接触穿透；不会用保存的物体姿态驱动回放。', '',
'这些是验证成功的初始示范样本，不保证最短路径。采集尝试通过率不是独立测试集上的策略成功率；正式训练需增加初始状态数量并划分独立测试 seed。', '',
'![全部轨迹的最终场景](overview.png)', '',
'[四字母回放视频](planner_preview.mp4)', '',
'| 字母 | seed | 覆盖率 | 步数 | 验证 |', '|---|---:|---:|---:|---|'
]
lines += [f"| {x['letter']} | {x['seed']} | {100*x['final_coverage']:.2f}% | {x['steps']} | {'通过' if x['valid'] else '未通过'} |" for x in a]
(root/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:14]))
