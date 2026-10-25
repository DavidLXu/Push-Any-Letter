"""Read-only log watcher: persist reward and success curves without touching a learner."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import io
import json
import math
import time
from pathlib import Path


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_jsonl(path):
    if not path.exists():
        return []
    # A final unterminated line may still be in flight.
    text = path.read_text()
    complete = text[:text.rfind('\n')+1]
    return [json.loads(line) for line in complete.splitlines() if line.strip()]


def read_reward(path):
    if not path.exists():
        return []
    rows = []
    for row in csv.DictReader(io.StringIO(path.read_text())):
        try:
            step = int(row['time/total_timesteps'])
            reward = float(row['rollout/ep_rew_mean'])
            length = float(row['rollout/ep_len_mean'])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(reward) and math.isfinite(length):
            rows.append(dict(step=step, mean_episode_return=reward, mean_episode_length=length))
    return rows


def save_csv(path, rows, columns):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def render(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rewards = read_reward(run/'progress.csv')
    metrics = read_jsonl(run/'metrics.jsonl')
    evaluations = read_jsonl(run/'evaluations.jsonl')
    if not metrics:
        return dict(state='waiting_for_logs')
    out = run/'live_curves'
    out.mkdir(exist_ok=True)
    successes = []
    for metric in metrics:
        for key, series, count_key in [('train_success', 'training_mixed', 'episodes_window'),
                                      ('train_full_success', 'training_full_resets', 'full_episodes_window')]:
            if metric.get(key) is not None:
                successes.append(dict(step=metric['step'], series=series,
                    success_rate=metric[key], cases=metric[count_key]))
    for result in evaluations:
        successes.append(dict(step=result['step'], series=f"evaluation_{result['cases']}",
                              success_rate=result['success_rate'], cases=result['cases']))
    if rewards:
        save_csv(out/'reward_history.csv', rewards, ['step','mean_episode_return','mean_episode_length'])
    save_csv(out/'success_history.csv', successes, ['step','series','success_rate','cases'])
    plt.rcParams.update({'figure.facecolor':'#f1f4ef','axes.facecolor':'#f1f4ef',
        'text.color':'#172b31','axes.labelcolor':'#172b31', 'axes.spines.top':False,'axes.spines.right':False})
    fig, (left,right) = plt.subplots(1,2,figsize=(13,4.8))
    if rewards:
        left.plot([r['step']/1e6 for r in rewards],[r['mean_episode_return'] for r in rewards], color='#e7783e')
    left.set_title('Reward · completed episode return',loc='left')
    left.set_ylabel('Mean shaped return · latest 100 episodes')
    definitions = [('training_mixed','Training · mixed resets (rolling 1,000)', '#e7783e', '-'),
                   ('training_full_resets','Training · full-reset subset', '#8b746c', '-'),
                   ('evaluation_52','Independent monitoring · 52 cases', '#2b9a8f', 'o-'),
                   ('evaluation_520','Independent final test · 520 cases', '#172b31', 'D')]
    for series,label,color,style in definitions:
        rows=[r for r in successes if r['series']==series]
        if rows:
            right.plot([r['step']/1e6 for r in rows],[100*r['success_rate'] for r in rows], style, color=color,label=label,markersize=5)
    right.set_title('Success rate · distinct evaluation scopes',loc='left')
    right.set_ylabel('Successful episodes (%)')
    right.set_ylim(-1,100)
    right.legend(fontsize=8,loc='upper left')
    for ax in (left,right):
        ax.set_xlabel('Online environment transitions (millions)')
        ax.grid(alpha=.16)
    fig.text(.02,.025,'Reward includes potential shaping, success bonus, and time/control costs. Curriculum changes during training.\nOnly completed evaluations are plotted; missing evaluation points are not zero.',fontsize=8,color='#536d65')
    fig.tight_layout(rect=(0,.09,1,1))
    for suffix in ('png','pdf'):
        tmp = out/f'.reward_success.tmp.{suffix}'
        fig.savefig(tmp,dpi=170)
        tmp.replace(out/f'reward_success.{suffix}')
    plt.close(fig)
    return dict(state='running', step=metrics[-1]['step'], reward_points=len(rewards),
                success_points=len(successes), evaluations=len(evaluations))


def main():
    import fcntl
    import traceback
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--interval',type=float,default=30)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    run=Path(args.run)
    run.mkdir(parents=True,exist_ok=True)
    with (run/'curve_service.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                result=render(run)
                training=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
                terminal=training.get('status') in ('complete','failed')
                if terminal:
                    result['state']='complete' if training['status']=='complete' else 'training_failed'
                result.update(pid=os.getpid(),updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                atomic_json(run/'curve_service_status.json',result)
                print(json.dumps(result),flush=True)
                if args.once or terminal:
                    return
            except BaseException:
                atomic_json(run/'curve_service_status.json',dict(state='failed',traceback=traceback.format_exc()))
                raise
            time.sleep(max(5,args.interval))


if __name__=='__main__':
    main()
