# It Is the Operating Point, Not the Ranking

Code and data for the paper "It Is the Operating Point, Not the Ranking: A
Confound in Cross-Domain Recall Comparison for Abusive-Language Detection"
(manuscript in preparation).

Twelve training runs: four configurations x three seeds, on an identical
data split. Evaluated on a held-out partition and on 2,834 human-annotated
YouTube comments from Muminovic (2025), IJCA 187(25).

## Reproducing the analysis
    pip install -r code/requirements.txt
    cd code
    python agreement_analysis.py
    python make_figures.py

Run both scripts from inside `code/`. They locate `data/` and `predictions/`
relative to their own location, so this works regardless of where the repo
is cloned.

`agreement_analysis.py` regenerates every reported statistic on the YouTube
corpus (the flag-count analysis, run-to-run agreement, coverage and majority
voting) from the twelve prediction files. `make_figures.py` regenerates both figures.

## Retraining
`code/gemma-notebook.ipynb` runs on Kaggle with a GPU. It will not run
locally or in a plain Jupyter install — it depends on Kaggle-specific
features (attached input datasets, Kaggle Secrets, a GPU runtime). The
notebook is published without cell outputs; results are in `data/` and
`predictions/`.

Before running it on your own Kaggle account, attach:
- **Kaggle Input datasets:** the HateXplain CSV, the contrastive-examples
  CSV, the HateCheck test suite, and the Muminovic YouTube corpus. The
  notebook searches for each by filename under `/kaggle/input/`, so the
  dataset slug you attach it under does not need to match ours.
- **Kaggle Model:** `google/gemma-2` (`gemma-2-2b-it` variant) — needed for
  the LLM-routing cells.
- **Kaggle Secret** (optional, only for the live-demo cell at the end):
  an `ngrok_token` secret from your own free ngrok account, used to expose
  the Flask demo app. Skip this if you only want the training/evaluation
  results.

Also change `MODEL_REPO` near the top of the configuration cell to a
Hugging Face repo under your own account, so the notebook pushes/pulls
checkpoints from a repo you control rather than ours.

Set `RUN_LABEL` and `TRAIN_SEED` at the top of the configuration cell for
each of the twelve runs. Approx. 3.5 h per run.

## What is and is not included
Included: our model predictions (row index, gold label, binary prediction),
run metrics, analysis code, and the training notebook (which also contains
the corpus-construction code).
Not included: the text of the evaluation corpus or any training corpus.
These are available from their original authors under their own terms.

Predictions are keyed by `row_id`, the zero-based row index into the
evaluation corpus as distributed by Muminovic (2025). Join on that index to
recover the comment text from the original source.

`predictions/checksums.txt` gives SHA-256 hashes for each prediction file.
To check them after cloning:

    cd predictions
    sha256sum -c checksums.txt            # Linux / macOS (shasum -a 256 -c on older macOS)

On Windows (PowerShell), compare the output of
`Get-FileHash second_corpus_main_0.csv` with the matching line; PowerShell
prints the hash in capitals, which is the same value.

## Licence
MIT (code and our predictions). Source corpora retain their own licences.

## Citing
If you use this code or these predictions, please cite the paper
(reference to be added on publication).
