# SPHERE query investigation: reproduction

Read-only exploratory scripts. They write nothing into the repo data directories.
All outputs go to a scratch directory. Run from this directory.

Setup used (scratch venv with torch, transformers, pyarrow; the repo venv has no torch):

    export W=<scratch dir>
    export PY=<scratch venv>/bin/python
    export PYTHONPATH=.:<repo root>

Data sources (HF datasets): sjoon-oh/aker (sphere-10m.npy, spacev-10m.npy,
triviaqa-sim-100k-*.npy, spacev-sim-100k-*.npy, triviaqa.txt) and mohdumar/SPHERE_100K
(Hermes SPHERE_100K parquet: id, url, title, sha, raw, vector). Sphere checkpoint:
http://dl.fbaipublicfiles.com/sphere/dpr_web_biencoder.cp (2.6 GB pickle, untrusted).

## Part A

    # 1. sphere-10m rows (float64): representability, norms, mean, outlier dims
    $PY 01_inspect_sphere_rows.py 2000 0

    # 2. map rows to Hermes passages (download train-00000-of-00002.parquet first)
    curl -L -o $W/sphere100k_0.parquet \
      https://huggingface.co/datasets/mohdumar/SPHERE_100K/resolve/main/data/train-00000-of-00002.parquet
    $PY 02_map_rows_to_passages.py $W/sphere100k_0.parquet 2000

    # 3. pickle -> safetensors in an isolated venv, clean env, network denied
    curl -L -o $W/dpr_web_biencoder.cp http://dl.fbaipublicfiles.com/sphere/dpr_web_biencoder.cp
    python3 -m venv $W/iso/v && $W/iso/v/bin/pip install numpy safetensors
    env -i HOME=/var/empty PATH=/usr/bin:/bin /usr/bin/sandbox-exec \
      -p '(version 1)(allow default)(deny network*)' \
      $W/iso/v/bin/python -I 04_convert_dpr_web.py \
      $W/dpr_web_biencoder.cp $W/dpr_web.safetensors $W/report.json
    $PY 05_safetensors_to_hf.py $W/dpr_web.safetensors $W/hf_out

    # 4. candidates (rows 0..999) and the sweep; ROW_OFFSET=40000 gives held-out rows
    HF_HUB_OFFLINE=1 $PY 03_candidate_encoders.py $W/sphere100k_0.parquet 1000 \
      st:BAAI/bge-base-en-v1.5:raw:512 \
      dpr_ctx:facebook/dpr-ctx_encoder-multiset-base:pair:256 \
      dpr_ctx:$W/hf_out/ctx:pair:256 dpr_ctx:$W/hf_out/ctx:pair_tt0:256
    ROW_OFFSET=40000 HF_HUB_OFFLINE=1 $PY 03_candidate_encoders.py \
      $W/sphere100k_0.parquet 500 dpr_ctx:$W/hf_out/ctx:pair_tt0:256

Candidate spec: `kind:model:fmt:max_len[:pool]` (see the script docstring).

## Part B (10 minute cap)

    $PY -u 06_partb_download.py $W/partb 100 1000      # probe, estimate, slices
    for s in train validation test; do curl -L -o $W/partb/tqa_$s.parquet \
      "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/refs%2Fconvert%2Fparquet/rc.nocontext/$s/0000.parquet"; done
    HF_HUB_OFFLINE=1 $PY -u 07_partb_metrics.py $W/partb $W/sphere100k_0.parquet \
      $W/hf_out/question $W/partb_results.json 3000
    # lift and stream skew only (what the second run did):
    HF_HUB_OFFLINE=1 SKIP_METRICS=1 $PY -u 07_partb_metrics.py $W/partb \
      $W/sphere100k_0.parquet $W/hf_out/question $W/partb_results_lift.json 3000

## Addenda (outside the Part B cap)

    HF_HUB_OFFLINE=1 $PY 08_aker_query_alignment.py $W/partb 200
    HF_HUB_OFFLINE=1 $PY 09_encoder_throughput.py $W/partb $W/hf_out/question 1000

`common.py` (range reads, cosine) and `metrics.py` (relative contrast, hubness, norms,
stream skew) are helpers. Note: the Part B script as committed reflects the second run;
the first run (all metrics, no lift) printed the table values quoted in REPORT.md.
