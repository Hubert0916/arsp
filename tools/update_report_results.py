"""Generate report numbers and analysis directly from a complete scored run."""
import argparse
import json
from pathlib import Path
import shutil

from prepare_proxy import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "results/proxy")
    args = parser.parse_args()
    result = json.loads((args.directory / "summary.json").read_text())
    manifest = json.loads((ROOT / "experiments/proxy_manifest.json").read_text())
    if result["n_groups"] != len(manifest["groups"]) or result["n_rows"] != sum(len(g["evaluation"]) for g in manifest["groups"]):
        raise ValueError("Only the complete published split may update the report")
    shutil.copyfile(args.directory / "summary.json", ROOT / "report/results_proxy.json")
    a, b = result["asrp_brier"], result["empirical_mean_brier"]
    aa, ba = result["asrp_alc"], result["empirical_mean_alc"]
    def texrow(name, values, bold=False):
        cells = [f"{x:.4f}" for x in values]
        if bold:
            cells = ["\\textbf{"+v+"}" for v in cells]
        return name + " & " + " & ".join(cells) + "\\\\\n"
    table = texrow(r"\textbf{ASRP}", a+[aa])
    table += texrow("Empirical Mean", b+[ba])
    table += texrow("Difference", [x-y for x, y in zip(a+[aa], b+[ba])])
    table = (r"\begin{tabular}{lrrrrrrr}" + "\n" + r"\toprule" + "\n"
             + r"Method & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ & ALC\\" + "\n"
             + r"\midrule" + "\n" + table + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    (ROOT / "report/results_table.tex").write_text(table)
    gains = [y-x for x, y in zip(a, b)]
    wins = sum(g > 0 for g in gains)
    j = max(range(6), key=lambda i: gains[i])
    reduction = ba-aa
    win_text = "all six budgets" if wins == 6 else f"{wins} of the six budgets"
    overview = (rf"Table~\ref{{tab:proxy}} compares the two predictors on identical evaluation items and acquired labels. "
                rf"ASRP has lower aggregate Brier scores at \textbf{{{win_text}}}. "
                rf"Its Brier ALC is ${{{aa:.4f}}}$, compared with ${{{ba:.4f}}}$ for Empirical Mean, "
                rf"an absolute {'reduction' if reduction >= 0 else 'increase'} of ${{{abs(reduction):.4f}}}$ "
                rf"(${{{abs(reduction/ba)*100:.1f}\%}}$ relative to the baseline). "
                r"Differences use unrounded scores; negative values favor ASRP. These are local public-proxy results.")
    (ROOT / "report/results_overview.tex").write_text(overview+"\n")
    analysis = (r"\paragraph{Where the improvement occurs.}"+"\n"
        +rf"The largest baseline-minus-ASRP gap occurs at \textbf{{budget {result['budgets'][j]}}}: "
        +rf"ASRP scores ${a[j]:.4f}$ and Empirical Mean scores ${b[j]:.4f}$. "
        +r"At budget one, Empirical Mean predicts either zero or one for every query in a group; a single unrepresentative label can therefore produce a large error. "
        +r"ASRP retains its public prior and Rasch component at this budget, moderating such extreme updates. "
        +rf"At budget zero, the content- and prior-based ASRP prediction scores ${a[0]:.4f}$, compared with the constant baseline's $0.2500$."
        +"\n\n"+r"\paragraph{Behavior as evidence accumulates.}"+"\n"
        +rf"ASRP moves from ${a[0]:.4f}$ at budget zero to ${a[-1]:.4f}$ at budget 31. "
        +rf"The baseline-minus-ASRP gaps at budgets 15 and 31 are ${gains[-2]:.4f}$ and ${gains[-1]:.4f}$. "
        +r"The budget-dependent design anchors early predictions with the Rasch hybrid and gives the kernel residual more weight as support grows. "
        +r"Topic calibration remains active, but the source-name guard has zero weight adjustment for the anonymous aliases used in this experiment. "
        +r"These results compare the combined predictors conditional on matched evidence; the constrained acquisition trajectory does not measure an active-selection gain."
        +"\n")
    (ROOT / "report/results_analysis.tex").write_text(analysis)
    ending = (r"In the local public proxy, ASRP achieves lower aggregate Brier scores than Empirical Mean at "
              + ("all six budgets." if wins == 6 else f"{wins} of the six budgets."))
    (ROOT / "report/results_conclusion.tex").write_text(ending+"\n")
    markdown = ["| Method | B0 | B1 | B3 | B7 | B15 | B31 | Brier ALC |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, values in [("ASRP", a+[aa]), ("Empirical Mean", b+[ba])]:
        markdown.append("| " + name + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    readme = ROOT / "README.md"
    prefix, remainder = readme.read_text().split("<!-- RESULTS_START -->", 1)
    _, suffix = remainder.split("<!-- RESULTS_END -->", 1)
    readme.write_text(prefix + "<!-- RESULTS_START -->\n\n" + "\n".join(markdown)
                      + "\n\n<!-- RESULTS_END -->" + suffix)
    print(f"Updated report inputs from {args.directory / 'summary.json'}")


if __name__ == "__main__":
    main()
