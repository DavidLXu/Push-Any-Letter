"""Render only recorded points; training and independent evaluation are distinct."""
import json
from pathlib import Path


def plot_run(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(directory)
    metrics = [json.loads(x) for x in (out/"metrics.jsonl").read_text().splitlines()]
    evalpath = out/"evaluations.jsonl"
    evaluations = [json.loads(x) for x in evalpath.read_text().splitlines()] if evalpath.exists() else []
    plt.rcParams.update({"figure.facecolor":"#f1f4ef", "axes.facecolor":"#f1f4ef",
        "text.color":"#172b31", "axes.labelcolor":"#172b31", "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1, 3, figsize=(16,4.5))
    for key, label, color in [("train/value_loss","Value loss","#e7783e"),
                               ("train/policy_gradient_loss","Policy loss","#2b9a8f")]:
        rows = [r for r in metrics if key in r]
        axes[0].plot([r["step"] for r in rows], [r[key] for r in rows],label=label,color=color)
    for key, label, color in [("train_success","Mixed reset training (rolling 1,000 episodes)","#e7783e"),
                             ("train_full_success","Full reset training subset","#8b746c")]:
        rows = [r for r in metrics if r.get(key) is not None]
        axes[1].plot([r["step"] for r in rows], [100*r[key] for r in rows],label=label,color=color)
    for cases, label, color in [(52,"Fixed 52-case evaluation","#2b9a8f"),(520,"Independent final 520 cases","#172b31")]:
        rows = [r for r in evaluations if r["cases"]==cases]
        axes[1].plot([r["step"] for r in rows],[100*r["success_rate"] for r in rows],"o-",label=label,color=color)
    rows = [r for r in metrics if "train/approx_kl" in r]
    axes[2].plot([r["step"] for r in rows],[r["train/approx_kl"] for r in rows],color="#567c91",label="Approximate KL")
    for ax, title in zip(axes,["Optimization losses", "Success rate (%)", "PPO update size"]):
        ax.set_title(title,loc="left"); ax.set_xlabel("Online environment transitions"); ax.grid(alpha=.15)
        if ax.lines: ax.legend(fontsize=7)
    axes[1].set_ylim(0,100)
    fig.tight_layout()
    fig.savefig(out/"curves.png",dpi=170)
    fig.savefig(out/"curves.pdf")
    plt.close(fig)
