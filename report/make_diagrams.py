"""Draw the architecture diagram and the tool-flow diagram, and crop screenshots for the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

A = Path(__file__).parent / "assets"
NAVY, BLUE, GREEN, ORANGE, GREY, RED = "#1F3A5F", "#2F6DB5", "#2E7D5B", "#C77A1A", "#EEF1F5", "#B03A48"


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, color="white", bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p1, p2, text=None, color="#333333", rad=0.0, fs=8.5, tdx=0.0, tdy=0.12, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, lw=1.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p1[0] + p2[0]) / 2 + tdx, (p1[1] + p2[1]) / 2 + tdy, text, ha="center", va="center",
                fontsize=fs, color=color, style="italic")


def architecture():
    fig, ax = plt.subplots(figsize=(11.4, 7.4))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    box(ax, 0.3, 6.2, 3.3, 0.9, "User\nStreamlit app / CLI", GREY, NAVY, color=NAVY)
    box(ax, 7.2, 6.2, 3.9, 0.9, "Final output\nrecommendation + charts + trace", GREY, NAVY, color=NAVY, fs=9.5)
    # Orchestrator container
    ax.add_patch(FancyBboxPatch((0.3, 2.55), 6.6, 3.2, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#F7F9FC", ec=BLUE, lw=2))
    ax.text(0.5, 5.5, "Orchestrator Agent  (LangGraph ReAct loop)", fontsize=11.5, color=BLUE,
            fontweight="bold", va="center")
    box(ax, 0.6, 4.05, 2.85, 1.1, "LLM (reasoning)\nGemini / Groq / OpenRouter\nfallback chain", BLUE, fs=9)
    box(ax, 3.65, 4.05, 3.0, 1.1, "Planner\nsystem prompt: profile ->\npreprocess -> compare -> tune",
        GREEN, fs=8.5)
    box(ax, 0.6, 2.8, 2.85, 0.95, "Memory\nconversation context\n(LangGraph MemorySaver)", ORANGE, fs=8.5)
    box(ax, 3.65, 2.8, 3.0, 0.95, "Observability\nJSONL trace logger\n(thought / action / observation)",
        NAVY, fs=8.5)
    # Tool container
    ax.add_patch(FancyBboxPatch((7.7, 0.45), 3.4, 5.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="#FBF7F0", ec=ORANGE, lw=2))
    ax.text(9.4, 5.2, "Tool APIs", fontsize=11.5, color=ORANGE, fontweight="bold", ha="center")
    for i, t in enumerate(["profile_dataset", "preprocess_data", "compare_models",
                           "analyze_best_model", "calculator"]):
        box(ax, 7.95, 4.35 - i * 0.82, 2.9, 0.62, t, "white", ORANGE, fs=9.5, color="#7A4A0A")
    # Data layer
    box(ax, 0.3, 0.45, 3.0, 1.35, "Data workspace\nCSV datasets, pandas\nDataFrame, X / y state",
        GREY, NAVY, color=NAVY, fs=9)
    box(ax, 3.8, 0.45, 3.1, 1.35, "scikit-learn\nmodels, cross-validation,\ngrid search, permutation\nimportance",
        GREY, NAVY, color=NAVY, fs=9)
    # Arrows
    arrow(ax, (2.0, 6.2), (2.0, 5.78), "goal / question", tdx=1.0, tdy=0.0)
    arrow(ax, (5.6, 5.78), (7.2, 6.55), "answer", rad=-0.25, tdx=-0.35, tdy=0.35)
    arrow(ax, (6.9, 4.2), (7.7, 4.2), None, style="<|-|>")
    ax.text(7.3, 4.62, "call /\nobserve", fontsize=7.5, ha="center", va="center", color="#333",
            style="italic", linespacing=1.2)
    arrow(ax, (3.3, 1.1), (3.8, 1.1), None, style="<|-|>")
    arrow(ax, (7.7, 1.1), (6.9, 1.1), None)
    ax.text(5.7, 0.15, "tools read and update the data workspace; scikit-learn does the numeric work",
            fontsize=8.5, ha="center", color="#555", style="italic")
    ax.set_title("Architecture of the ML Model Advisor Agent", fontsize=13, color=NAVY, fontweight="bold", pad=8)
    fig.savefig(A / "diagram_architecture.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def toolflow():
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 11.9)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    box(ax, 0.15, 3.9, 1.5, 0.9, "User goal\n+ dataset path", GREY, NAVY, color=NAVY, fs=9)
    steps = [("1. profile_dataset", "rows, dtypes,\nmissing, unique"),
             ("2. preprocess_data", "target, drop IDs,\nimpute, encode"),
             ("3. compare_models", "5-fold CV for\n4-5 models"),
             ("4. analyze_best_model", "tune winner +\nfeature importance")]
    xs = [2.0, 4.5, 7.0, 9.5]
    for (name, sub), x in zip(steps, xs):
        box(ax, x, 3.9, 2.15, 0.9, name, ORANGE, fs=9)
        ax.text(x + 1.075, 3.65, sub, ha="center", va="top", fontsize=8, color="#444", linespacing=1.3)
    arrow(ax, (1.65, 4.35), (2.0, 4.35))
    for x in xs[:-1]:
        arrow(ax, (x + 2.15, 4.35), (x + 2.5, 4.35))
    # Reasoning loop spans under all four tools
    box(ax, 2.0, 1.55, 9.65, 1.0,
        "LLM reasoning (ReAct loop)\nThought -> chooses next Action -> reads Observation -> decides again", BLUE, fs=10)
    for x in xs:
        arrow(ax, (x + 1.075, 3.15), (x + 1.075, 2.55), None, color=BLUE, style="<|-|>")
    box(ax, 0.15, 1.55, 1.6, 1.0, "Memory\n(message history)", ORANGE, fs=8.5)
    arrow(ax, (1.75, 2.05), (2.0, 2.05), None, color=ORANGE, style="<|-|>")
    # Error branch and final answer
    box(ax, 8.9, 0.15, 2.75, 0.85, "Tool error: read message,\nfix arguments, retry", RED, fs=8.5)
    arrow(ax, (10.275, 1.55), (10.275, 1.0), "on error", color=RED, style="<|-|>", tdx=0.55, tdy=0.0)
    box(ax, 3.9, 0.15, 3.0, 0.85, "Final answer", GREEN, fs=11)
    arrow(ax, (5.4, 1.55), (5.4, 1.0), "enough evidence", tdx=1.05, tdy=0.0, color=GREEN)
    ax.text(5.95, 5.3, "Tool flow:  Goal -> Plan -> Tool use -> Observation -> Final output", ha="center",
            fontsize=12.5, color=NAVY, fontweight="bold")
    fig.savefig(A / "diagram_toolflow.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def crops():
    im = Image.open(A / "shot3b_trace_expanded.png")  # 1410 x 3300
    im.crop((0, 0, im.width, 1085)).save(A / "fig_trace.png")             # reasoning trace only
    im.crop((0, 1085, im.width, 1880)).save(A / "fig_answer.png")         # banner + final answer
    im.crop((0, 1880, im.width, im.height)).save(A / "fig_charts.png")
    q, a = Image.open(A / "shot4b_followup_question.png"), Image.open(A / "shot4c_followup_answer.png")
    out = Image.new("RGB", (q.width, q.height + 12 + a.height), "white")
    out.paste(q, (0, 0))
    out.paste(a, (0, q.height + 12))
    out.save(A / "fig_followup.png")


if __name__ == "__main__":
    architecture()
    toolflow()
    crops()
    print("done")
