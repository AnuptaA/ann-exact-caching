# SPHERE query/data encoder investigation

Date 2026-11-10. Machine: macOS, 8 cores, 8 GiB. Scripts and commands: README.md.
Marks: [V] verified by a measurement here, [I] inferred, [U] not verified.

## Verdicts

- Identification of the sphere-10m base encoder: CONFIRMED by the stated thresholds.
  Sphere DPR_web context encoder (BERT-base, no projection, CLS = pooler_output),
  input (title, text) as a sentence pair, max 256 tokens, token_type_ids all zero.
  1,000 rows: median cos 0.9993, 5th percentile 0.9969. Held-out 500 rows: 0.9993 / 0.9964.
- Part B: no metric shows a pathology of Aker's released (bge) TriviaQA queries against
  the controls that a matched encoder would remove. The matched encoder is not better on
  hubness and about equal on relative contrast and Euclidean answer lift.

## Part A: identify the base encoder

Elapsed: about 25 min wall (09:44 to 10:09), inside the 30 min box. The main session
ran tests concurrently, so encode rates are pessimistic (load average up to 11).

| step | what | elapsed |
|---|---|---|
| 1 | sources read | about 2 min |
| 2 | row mapping (parquet download 151 MB in 8 s, compare 3 s) | about 2 min |
| 3 | vector inspection | 2 s compute |
| 6 | .cp download 2.6 GB in the background, restricted load + safetensors | download about 1.5 min, convert seconds |
| 4 | candidate encodes (1,000 rows each) and sweep | about 17 min |

### Sources (step 1)
- [V] HF dataset sjoon-oh/aker: file list only (no README/dataset card exists, 404).
  sphere-10m.npy is 61,440,000,128 bytes: header 128, dtype <f8, shape (10,000,000, 768).
  Also spacev-10m.npy (int8, 10M x 100), triviaqa-sim-100k-{0.3,0.6,0.99}.{npy,pkl},
  spacev-sim-100k-{...}.{npy,pkl}, triviaqa.txt.
- [V] Sphere repo README (github.com/facebookresearch/sphere, raw main README.md):
  DPR_web = dpr_web_biencoder.cp, config dpr_web_sphere.yaml. Config (fetched): encoder
  hf_bert, pretrained_model_cfg bert-base-uncased, projection_dim 0, sequence_length 256,
  do_lower_case True, special tokens [Q],[R],[F],[START_ENT],[END_ENT],[D] (not used here:
  the checkpoint vocabulary is 30,522, no extension).
- [V] HF dataset mohdumar/SPHERE_100K (the Hermes SPHERE_100K mirror the repo already
  uses): 2 parquet files of 50,000 rows, columns id, url, title, sha, raw, vector
  (float64 list). Only file 0 was downloaded. `raw` is title + " " + text (995 of 1,000
  rows start with the title). Hermes paper/repo/cards were not read [U], nor the Aker
  repo and paper [U] (time box). The Aker paper's BGE-large claim was not re-checked.

### Row mapping and vector inspection (steps 2, 3)
- [V] Rows 0..1,999 of sphere-10m equal rows 0..1,999 of Hermes SPHERE_100K file 0
  (cosine 1.0000, max abs diff 5e-9 on row 0; nearest-vector match index equals row
  index for 200/200). So sphere-10m starts with Hermes rows, and passage text exists
  for rows 0..49,999 [V for 0..1,999, I beyond]. File 1 was not checked [U].
- [V] 2,000 rows: values are not exactly float32 or float16 representable (decimal-
  rounded to about 8 places). Norm min/p5/median/p95/max 13.82/14.06/14.40/14.81/15.39.
  Mean vector norm 12.8; cosine of a row to the mean vector 0.89 (median). One extreme
  dimension: dim 308 has mean -9.14 (all other dim means below 1.8 in magnitude).

### Candidates (step 4), 1,000 rows from Hermes file 0 vs the stored vectors
| encoder and input | cos p5 / median / p95 | rel L2 median |
|---|---|---|
| BAAI/bge-base-en-v1.5, CLS, raw text, 512 | 0.503 / 0.554 / 0.593 | 0.998 |
| facebook DPR ctx multiset, pair, 256 | 0.641 / 0.709 / 0.746 | 0.710 |
| Sphere DPR_web ctx, pair, 256, HF token types | 0.995 / 0.998 / 0.999 | 0.067 |
| Sphere DPR_web ctx, pair, 256, token types zero | 0.997 / 0.9993 / 0.9995 | 0.038 |

Not run in the box (skipped for time): bge-base-en, bert-base-uncased, DPR single-nq
ctx; the earlier brief's 0.56-0.59 for bge variants is consistent with the bge row [V
for v1.5 only]. 300-row sweep of the DPR_web ctx encoder (cos p5 / median):
pair 0.9948/0.9978; pair with the full raw text as second segment 0.9186/0.9519; raw
text as one sequence 0.9842/0.9915; "title. text" 0.9848/0.9925; pair at max 512
0.9948/0.9978 (same as 256); pair with zero token types 0.9967/0.9993; the same in
float16 0.9967/0.9993 (precision is not the cause). "pair" = tokenizer(title, text)
where text is raw without its title prefix.

Held-out (rows 40,000..40,499, not used for tuning): p5 0.9964, median 0.9993, p95 0.9995.

### Status: confirmed
Rule: confirmed = median >= 0.999 and p5 >= 0.99 with exact preprocessing recorded.
Median 0.9993 (1,000 rows and held-out 500) and p5 0.9969 / 0.9964 meet it. Recorded
preprocessing: bert-base-uncased tokenizer; input (title, raw minus the leading title
and one space); truncation to 256 tokens (longest_first); token_type_ids forced to 0
(the original DPR trains with all-zero segment ids; I recalled this, I did not check
the DPR source [I]); DPR_web ctx weights from the .cp; output = first-token hidden
state (pooler_output of HF DPRContextEncoder with projection_dim 0); no normalization;
fp32. Caveats [V]: the fit is not exact: median relative L2 0.038, minimum cosine 0.944
(1,000 rows). The residual is unexplained (not length, not float16). The rows that
fall below 0.99 were not inspected. The sweep chose the format on the same first 300
to 1,000 rows (held-out rows gave the same numbers).

### Pickle handling (what was and was not enforced)
- Downloaded only; never opened in the normal environment. Loaded by
  04_convert_dpr_web.py in a new venv (numpy, safetensors only; no torch, no
  transformers) with `env -i HOME=/var/empty PATH=/usr/bin:/bin`, python -I, no
  credentials, no trust_remote_code, under `sandbox-exec` with `(deny network*)`.
- Enforced [V]: network denial (an outbound connect inside the sandbox raised
  PermissionError, recorded by the script). Allowlist unpickler: OrderedDict, torch
  storage classes (as numpy dtypes) and torch._utils._rebuild_tensor_v2 only; storages
  read from the zip by my own persistent_load; every other global became an inert stub
  type (stubbed: omegaconf.*, typing.Any, collections.defaultdict, python-2 builtin
  names). Only the 400 tensors of model_dict were converted (ctx_model, question_model).
- Not enforced: no filesystem restriction (the sandbox profile was allow-default plus
  deny-network, so reads of any file were possible); no memory or CPU limits; venv
  creation and pip install ran before, with network. Afterwards only the safetensors
  were used (loaded by transformers in the scratch venv).
- The unconverted optimizer and scheduler state was not read.
- The Aker .pkl files were not downloaded or inspected [U] (time cap).

## Part B: query/data mismatch

Cap 10 min wall for download plus compute: used 592 s (9.9 min), including 100 s lost
to a failed relaunch (shell noclobber on the log file). Probe: a single 50 MB range
read took 2.2 s (22 MB/s); printed estimate 815 MB in 37 s; actual download 197 s,
because the stratified 6 MB block reads ran at about 4 MB/s. The estimate was too
optimistic by 5x. The Hermes file (151 MB) and the TriviaQA parquets (64 MB) were
downloaded before/besides and are not in the cap's byte count.

Sample sizes. Base slices: sphere-10m 100,000 rows = 100 evenly spaced blocks of 1,000
(float64 cast to float32); spacev-10m the same 100 x 1,000 rows (int8 cast to float32).
Slice values are not 10M values: relative contrast, hubness and nearest-neighbor
distances all depend on the base size; compare only within this table.
Queries: 3,000 per set, distinct vectors. Euclidean distance, exact brute force in
float32 (no released ground truth used; the Aker .pkl files were not opened).

Pairing caveat [V]: stream row i of triviaqa-sim-100k-0.6.npy is NOT line i of
triviaqa.txt (best-match rows are scattered; 200/200 lines match some row at cosine
0.9999995, so the vectors are exactly bge-base-en-v1.5 raw CLS [V], but not in line
order). The matched-encoder queries are DPR_web encodings of 3,000 lines of
triviaqa.txt (evenly spaced over its first 82,069 lines), a sample of the same question
pool but not the same questions as the Aker vector rows.

### Metrics b and c (10-NN, 100,000 base rows, 3,000 queries)
| base / queries | rel. contrast | hubness skew | top-100 share | median NN cosine |
|---|---|---|---|---|
| sphere / Aker TriviaQA vectors (bge) | 0.923 | 12.2 | 0.076 | 0.453 |
| sphere / DPR_web matched queries | 0.898 | 40.0 | 0.162 | 0.570 |
| sphere / held-out passages (rows 10,000..12,999 of Hermes) | 0.728 | 8.19 | 0.048 | 0.908 |
| spacev / Aker SPACEV stream (0.6) | 0.727 | 5.13 | 0.031 | 0.630 |
| spacev / held-out spacev-10m rows 10,000..12,999 | 0.693 | 22.9 | 0.088 | 0.652 |

Relative contrast = median over queries of the 10th-NN distance divided by the median
distance to the base (as specified). Smaller means a more distinct nearest neighbor.
Hubness skew = skewness of the 10-occurrence counts over the 100,000 base rows;
top-100 share = share of all 30,000 top-10 occurrences held by the 100 most frequent
vectors. These numbers come from the first run of 07_partb_metrics.py (log kept in
this report only; the JSON of that run was not written because the lift stage failed
after it, see deviations).

### Metric a: answer-passage lift, Euclidean (300 TriviaQA validation questions, 50,000 Hermes passages)
| encoder | top-20 precision | lift | any hit |
|---|---|---|---|
| bge-base-en-v1.5 raw CLS | 0.041 | 11.7x | 0.273 |
| dpr_web question encoder | 0.043 | 12.3x | 0.287 |
| dpr single-nq question encoder | 0.013 | 3.7x | 0.130 |

Inner product, same set: bge 13.0x, dpr_web 19.7x, single-nq 13.4x. [V] This reproduces
workloads/q1/datasets/triviaqa_encoder_evidence.md exactly (11.7x, 12.3x, 3.7x
Euclidean; 13.0x, 19.7x, 13.4x inner product). The dpr_web encoder here is my converted
question_model from the same .cp, not the repo's dpr_web_query_encoder_hf tarball;
the identical numbers suggest the weights agree [I]. Lift for Aker's own released
vectors is not computable: Aker's triviaqa.txt lines are generated paraphrases
(only 8 of the first 100,000 normalize to a TriviaQA question) with no answer aliases.
The first 300 validation questions are not the same 300 used before, yet the numbers
are identical to the third digit, so the earlier file likely used the same first 300 [I].

### Metric d: effective skew of released streams (by exact vector identity)
| stream | requests | distinct | top-1 vector share | top-10 share | top 1% of distinct share |
|---|---|---|---|---|---|
| triviaqa 0.6 | 99,990 | 81,217 | 0.00038 | 0.0027 | 0.058 |
| spacev 0.3 | 99,914 | 97,941 | 0.00004 | 0.0003 | 0.020 |
| spacev 0.6 | 99,982 | 94,447 | 0.00013 | 0.0011 | 0.033 |
| spacev 0.99 | 99,852 | 65,289 | 0.00095 | 0.0087 | 0.201 |

Group labels (from the .pkl) were not used [U]; only the 0.6 TriviaQA stream was
downloaded. At the same nominal parameter 0.6 the TriviaQA stream is a bit more skewed
than the SPACEV one (5.8% vs 3.3%).

### Is any metric pathological versus the controls? No.
- Relative contrast: Aker bge queries 0.923 vs matched DPR_web queries 0.898 (similar,
  0.025 apart). Both are above the in-space controls (0.73, 0.69), but the matched
  encoder shows the same gap, so it is a query-to-passage property (questions are not
  passages), not encoder mismatch.
- Hubness: the bge queries (skew 12.2, top-100 share 7.6%) are less hubby than the
  matched queries (40.0, 16.2%) and in the range of the controls (5.1 to 22.9, 3.1% to
  8.8%). The matched encoder is worse, so a matched encoder would not remove it.
- Answer lift (Euclidean): bge 11.7x vs matched 12.3x, equal within noise (300
  questions). Matched wins only under inner product, which Q1 does not use.
- Median NN cosine is lower for query sets (0.45, 0.57) than for held-out passages
  (0.91); again true for both encoders.
Caveats: 3,000 queries and a 100,000-row stratified slice, not 10M; one stream
parameter; Aker vector rows and the matched queries are different questions.
