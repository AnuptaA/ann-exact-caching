#!/bin/sh
# MedCPT article embeddings of PubMed (NCBI: https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/):
# 38 chunks of ~1M articles, (N, 768) float32 + the PMID list; rate-limited (quiet timing guard: <= 50 MB/s)
cd "$(dirname "$0")/pubmed_medcpt" || exit 1
U=https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings
curl -sSL -o README.txt $U/README.txt
for i in $(seq 0 37); do
  for f in pmids_chunk_$i.json embeds_chunk_$i.npy; do
    [ -f $f.ok ] && continue
    curl -sSL --retry 10 --limit-rate 15M -C - -o $f $U/$f && touch $f.ok || echo "FAILED $f"
  done
done
touch DONE
