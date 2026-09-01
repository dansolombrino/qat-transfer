"""Same-checkpoint contrast table: one encoder, one dataset, two readouts."""
import json, glob, os
import numpy as np

B = os.path.expanduser("~/qat-transfer/storage/checkpoints/text/sentence_transformers/same_checkpoint_contrast")

def main():
    rows = [json.load(open(f)) for f in
            glob.glob(os.path.join(B, "**", "contrast.json"), recursive=True)]
    rows.sort(key=lambda j: (j["dataset_name"], -j["bits"]))
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{\textbf{The asymmetry survives the strictest control.} One encoder "
         r"(Qwen3-Embedding-0.6B), one dataset, one quantization: the \emph{same} embeddings of "
         r"the test split are read out as classification, through a linear probe fitted on "
         r"full-precision training embeddings and then frozen, and as retrieval, by cosine "
         r"nearest neighbour within the split. Only the readout differs, so the gap between the "
         r"columns cannot be attributed to a different checkpoint, dataset, or objective. "
         r"Separation is the median $\gapk{2}/2\varepsilon$ of the corresponding scores.}",
         r"\label{tab:contrast}", r"\begin{tabular}{llcccc}", r"\toprule",
         r"Dataset & Bits & \multicolumn{2}{c}{Top-1 change rate} & "
         r"\multicolumn{2}{c}{Median separation} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r" & & Classification & Retrieval & Classification & Retrieval \\", r"\midrule"]
    for j in rows:
        L.append(f"{j['dataset_name']} & W{j['bits']} & {100*j['clsf_flip']:.1f}\\% & "
                 f"\\textbf{{{100*j['retr_flip']:.1f}\\%}} & {j['clsf_sep']:.2f} & "
                 f"{j['retr_sep']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.expanduser("~/qat-transfer/paper/tables/new_contrast.tex"), "w").write("\n".join(L) + "\n")
    r4 = [j for j in rows if j["bits"] == 4]
    print("wrote new_contrast.tex; W4 ratios: " +
          ", ".join(f"{j['retr_flip']/j['clsf_flip']:.1f}x" for j in r4))

if __name__ == "__main__":
    main()
