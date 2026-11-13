"""TripClick (gated, non-commercial; do not share or host any part): the benchmark documents
(1.52M MEDLINE title + abstract), the click logs (5.3M records, 2013-2020) and the topics, from
the Hugging Face dataset Navid/tripclick (needs a token with access in ~/.cache/huggingface/token)."""
from huggingface_hub import snapshot_download
path = snapshot_download("Navid/tripclick", repo_type="dataset",
                         local_dir="datasets/tripclick/raw", max_workers=4,
                         allow_patterns=["benchmark_tsv/documents/*", "benchmark_tsv/topics/*", "logs/*.json",
                                         "LICENSE.md"])
print("done:", path)
