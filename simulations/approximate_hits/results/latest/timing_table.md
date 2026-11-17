| trace | budget | technique | hit % | ret. hits | recall all | DiskANN ms | cache+DiskANN ms | hit ms | miss ms | p99 ms | DiskANN p99 | speedup | 1/(1-h) | env |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zipf 0.3 | 0.3% | Ours | 1.07 | 0.935 | 0.956 | 7.54 | 7.69 | 0.25 | 7.74 | 12.1 | 11.9 | 0.981 | 1.01 | contended |
| Zipf 0.3 | 1% | Ours (cutoff lost 79k requests) | 0 | - | 0.956 | 7.54 | 7.8 | - | 7.8 | 12.2 | 11.9 | 0.968 | 1 | contended |
| Zipf 0.3 | 3% | Ours (cutoff lost 79k requests) (cutoff lost 79k requests) | 0 | - | 0.956 | 7.54 | 7.82 | - | 7.82 | 12.2 | 11.9 | 0.965 | 1 | contended |
| Zipf 0.3 | 10% | Ours (cutoff lost 79k requests) (cutoff lost 79k requests) | 12.3 | 0.917 | 0.949 | 7.54 | 6.99 | 0.807 | 7.98 | 12.3 | 11.9 | 1.08 | 1.14 | contended |
| Zipf 0.3 | 0.3% | Ours (drift handling) (cutoff lost 37k requests) | 0.995 | 0.924 | 0.956 | 7.54 | 7.75 | 0.317 | 7.81 | 12.2 | 11.9 | 0.974 | 1.01 | contended |
| Zipf 0.3 | 1% | Ours (drift handling) (cutoff lost 37k requests) (cutoff lost 33k requests) | 1.69 | 0.936 | 0.955 | 7.54 | 7.73 | 0.437 | 7.87 | 12.3 | 11.9 | 0.976 | 1.02 | contended |
| Zipf 0.3 | 3% | Ours (drift handling) (cutoff lost 37k requests) (cutoff lost 33k requests) (cutoff lost 45k requests) | 2.01 | 0.929 | 0.955 | 7.54 | 7.71 | 0.494 | 7.85 | 12.3 | 11.9 | 0.979 | 1.02 | contended |
| Zipf 0.3 | 10% | Ours (drift handling) (cutoff lost 37k requests) (cutoff lost 33k requests) (cutoff lost 45k requests) | 11.4 | 0.926 | 0.95 | 7.54 | 7.1 | 0.885 | 8.02 | 12.3 | 11.9 | 1.06 | 1.11 | contended |
| Zipf 0.3 | 0.3% | Ours (continuous greedy) (cutoff lost 56k requests) | 0.415 | 0.916 | 0.956 | 7.54 | 7.73 | 0.275 | 7.77 | 12.2 | 11.9 | 0.976 | 1 | contended |
| Zipf 0.3 | 1% | Ours (continuous greedy) (cutoff lost 56k requests) (cutoff lost 43k requests) | 1.11 | 0.919 | 0.955 | 7.54 | 7.7 | 0.418 | 7.81 | 12.2 | 11.9 | 0.98 | 1.01 | contended |
| Zipf 0.3 | 3% | Ours (continuous greedy) (cutoff lost 56k requests) (cutoff lost 43k requests) (cutoff lost 1k requests) | 6.77 | 0.928 | 0.952 | 7.54 | 7.43 | 0.514 | 7.97 | 12.2 | 11.9 | 1.01 | 1.06 | contended |
| Zipf 0.3 | 10% | Ours (continuous greedy) (cutoff lost 56k requests) (cutoff lost 43k requests) (cutoff lost 1k requests) | 15.5 | 0.928 | 0.948 | 7.54 | 6.81 | 0.928 | 8.05 | 12.4 | 11.9 | 1.11 | 1.16 | contended |
| Zipf 0.3 | 0.3% | LRU (cutoff lost 38k requests) | 0.751 | 0.933 | 0.956 | 7.54 | 7.89 | 0.476 | 7.94 | 12.4 | 11.9 | 0.956 | 1.01 | contended |
| Zipf 0.3 | 1% | LRU (cutoff lost 38k requests) (cutoff lost 11k requests) | 2.9 | 0.93 | 0.955 | 7.54 | 7.98 | 0.866 | 8.2 | 12.6 | 11.9 | 0.945 | 1.03 | contended |
| Zipf 0.3 | 3% | LRU (cutoff lost 38k requests) (cutoff lost 11k requests) (cutoff lost 4k requests) | 6.46 | 0.928 | 0.953 | 7.54 | 7.97 | 1.12 | 8.43 | 12.8 | 11.9 | 0.946 | 1.06 | contended |
| Zipf 0.3 | 10% | LRU (cutoff lost 38k requests) (cutoff lost 11k requests) (cutoff lost 4k requests) | 22.1 | 0.912 | 0.942 | 7.54 | 7.45 | 2.02 | 9 | 13.4 | 11.9 | 1.01 | 1.25 | contended |
| Zipf 0.3 | 0.3% | Aker | 0.29 | 0.921 | 0.956 | 7.54 | 7.56 | 0.0208 | 7.57 | 12 | 11.9 | 0.998 | 1 | contended |
| Zipf 0.3 | 1% | Aker | 0.886 | 0.926 | 0.955 | 7.54 | 7.53 | 0.0319 | 7.59 | 12 | 11.9 | 1 | 1.01 | contended |
| Zipf 0.3 | 3% | Aker | 2.66 | 0.936 | 0.955 | 7.54 | 7.41 | 0.036 | 7.6 | 12 | 11.9 | 1.02 | 1.03 | contended |
| Zipf 0.3 | 10% | Aker | 7.05 | 0.899 | 0.949 | 7.54 | 7.08 | 0.0352 | 7.61 | 12 | 11.9 | 1.07 | 1.08 | contended |
| Zipf 0.3 | 1% | Aker (paper default) | 3.29 | 0.761 | 0.949 | 7.54 | 7.33 | 0.0368 | 7.59 | 11.9 | 11.9 | 1.03 | 1.03 | contended |
| Zipf 0.3 | 10% | Aker (paper default) | 9.03 | 0.812 | 0.94 | 7.54 | 6.94 | 0.033 | 7.59 | 11.9 | 11.9 | 1.09 | 1.1 | contended |
| Zipf 0.3 | 1% | QVCache (paper's D) | 5.47 | 0.34 | 0.922 | 7.54 | 7.79 | 0.644 | 8.2 | 12.5 | 11.9 | 0.968 | 1.06 | contended |
| Zipf 0.3 | 10% | QVCache (paper's D) | 5.5 | 0.954 | 0.955 | 7.54 | 7.58 | 0.45 | 7.98 | 12.5 | 11.9 | 0.996 | 1.06 | contended |
| Zipf 0.3 | 0.3% | QVCache | 0.204 | 0.939 | 0.956 | 7.54 | 8.28 | 0.696 | 8.29 | 12.7 | 11.9 | 0.911 | 1 | contended |
| Zipf 0.3 | 1% | QVCache | 1.38 | 0.878 | 0.955 | 7.54 | 8.64 | 1.12 | 8.7 | 13 | 11.9 | 0.873 | 1.01 | contended |
| Zipf 0.3 | 3% | QVCache | 3.98 | 0.922 | 0.954 | 7.54 | 8.61 | 1.38 | 8.91 | 13.3 | 11.9 | 0.877 | 1.04 | contended |
| Zipf 0.3 | 10% | QVCache | 5.55 | 0.946 | 0.954 | 7.54 | 7.41 | 0.255 | 7.81 | 12.2 | 11.9 | 1.02 | 1.06 | contended |
| Zipf 0.6 | 0.3% | Ours | 8.77 | 0.928 | 0.947 | 7.11 | 6.78 | 0.258 | 7.36 | 11.3 | 11.1 | 1.05 | 1.1 | contended |
| Zipf 0.6 | 1% | Ours | 10.8 | 0.926 | 0.946 | 7.11 | 6.73 | 0.331 | 7.44 | 11.4 | 11.1 | 1.06 | 1.12 | contended |
| Zipf 0.6 | 3% | Ours | 13.8 | 0.925 | 0.945 | 7.11 | 6.58 | 0.837 | 7.44 | 11.4 | 11.1 | 1.08 | 1.16 | contended |
| Zipf 0.6 | 10% | Ours | 20.6 | 0.921 | 0.942 | 7.11 | 6.3 | 1.13 | 7.6 | 11.5 | 11.1 | 1.13 | 1.26 | contended |
| Zipf 0.6 | 0.3% | Ours (drift handling) | 8.91 | 0.925 | 0.947 | 7.11 | 6.79 | 0.272 | 7.38 | 11.4 | 11.1 | 1.05 | 1.09 | contended |
| Zipf 0.6 | 1% | Ours (drift handling) | 10.5 | 0.928 | 0.947 | 7.11 | 6.78 | 0.381 | 7.46 | 11.5 | 11.1 | 1.05 | 1.1 | contended |
| Zipf 0.6 | 3% | Ours (drift handling) | 13.6 | 0.921 | 0.945 | 7.11 | 6.61 | 0.873 | 7.48 | 11.4 | 11.1 | 1.08 | 1.14 | contended |
| Zipf 0.6 | 10% | Ours (drift handling) | 21.3 | 0.913 | 0.94 | 7.11 | 6.31 | 1.23 | 7.65 | 11.5 | 11.1 | 1.13 | 1.24 | contended |
| Zipf 0.6 | 0.3% | Ours (continuous greedy) | 11.4 | 0.921 | 0.946 | 7.11 | 6.69 | 0.344 | 7.44 | 11.5 | 11.1 | 1.06 | 1.11 | contended |
| Zipf 0.6 | 1% | Ours (continuous greedy) | 13.9 | 0.926 | 0.945 | 7.11 | 6.58 | 0.432 | 7.49 | 11.4 | 11.1 | 1.08 | 1.14 | contended |
| Zipf 0.6 | 3% | Ours (continuous greedy) | 18.6 | 0.921 | 0.943 | 7.11 | 6.31 | 0.908 | 7.52 | 11.4 | 11.1 | 1.13 | 1.2 | contended |
| Zipf 0.6 | 10% | Ours (continuous greedy) | 27.6 | 0.918 | 0.938 | 7.11 | 5.94 | 1.29 | 7.76 | 11.5 | 11.1 | 1.2 | 1.33 | contended |
| Zipf 0.6 | 0.3% | LRU (cutoff lost 6k requests) | 4.28 | 0.937 | 0.95 | 7.11 | 7.23 | 0.464 | 7.49 | 11.5 | 11.1 | 0.984 | 1.04 | contended |
| Zipf 0.6 | 1% | LRU (cutoff lost 6k requests) (cutoff lost 1k requests) | 9.1 | 0.928 | 0.948 | 7.11 | 7.19 | 0.734 | 7.75 | 11.8 | 11.1 | 0.99 | 1.09 | contended |
| Zipf 0.6 | 3% | LRU (cutoff lost 6k requests) (cutoff lost 1k requests) | 14.6 | 0.924 | 0.945 | 7.11 | 7.05 | 1.52 | 7.96 | 12 | 11.1 | 1.01 | 1.15 | contended |
| Zipf 0.6 | 10% | LRU (cutoff lost 6k requests) (cutoff lost 1k requests) | 31.7 | 0.911 | 0.934 | 7.11 | 6.52 | 2.22 | 8.49 | 12.2 | 11.1 | 1.09 | 1.4 | contended |
| Zipf 0.6 | 0.3% | Aker | 1.57 | 0.885 | 0.95 | 7.11 | 7.04 | 0.0257 | 7.15 | 11.2 | 11.1 | 1.01 | 1.02 | contended |
| Zipf 0.6 | 1% | Aker | 3.76 | 0.89 | 0.948 | 7.11 | 6.96 | 0.0276 | 7.17 | 11.2 | 11.1 | 1.02 | 1.04 | contended |
| Zipf 0.6 | 3% | Aker | 8.13 | 0.909 | 0.945 | 7.11 | 6.59 | 0.0295 | 7.18 | 11.2 | 11.1 | 1.08 | 1.09 | contended |
| Zipf 0.6 | 10% | Aker | 12.6 | 0.918 | 0.942 | 7.11 | 6.32 | 0.0277 | 7.19 | 11.2 | 11.1 | 1.13 | 1.14 | contended |
| Zipf 0.6 | 1% | Aker (paper default) | 6.24 | 0.809 | 0.941 | 7.11 | 6.7 | 0.0324 | 7.18 | 11.2 | 11.1 | 1.06 | 1.07 | contended |
| Zipf 0.6 | 10% | Aker (paper default) | 13.9 | 0.86 | 0.934 | 7.11 | 6.16 | 0.0285 | 7.18 | 11.1 | 11.1 | 1.15 | 1.16 | contended |
| Zipf 0.6 | 1% | QVCache (paper's D) | 10.6 | 0.497 | 0.902 | 7.11 | 7.03 | 0.656 | 7.79 | 11.8 | 11.1 | 1.01 | 1.12 | contended |
| Zipf 0.6 | 10% | QVCache (paper's D) | 10.1 | 0.971 | 0.951 | 7.11 | 6.82 | 0.439 | 7.55 | 11.6 | 11.1 | 1.04 | 1.11 | contended |
| Zipf 0.6 | 0.3% | QVCache | 1.27 | 0.953 | 0.951 | 7.11 | 7.77 | 0.771 | 7.88 | 11.9 | 11.1 | 0.916 | 1.01 | contended |
| Zipf 0.6 | 1% | QVCache | 5.01 | 0.876 | 0.947 | 7.11 | 7.12 | 0.357 | 7.49 | 11.5 | 11.1 | 0.999 | 1.05 | contended |
| Zipf 0.6 | 3% | QVCache | 10.4 | 0.924 | 0.947 | 7.11 | 7.68 | 1.35 | 8.42 | 12.4 | 11.1 | 0.926 | 1.12 | contended |
| Zipf 0.6 | 10% | QVCache | 10.6 | 0.958 | 0.95 | 7.11 | 6.62 | 0.254 | 7.39 | 11.4 | 11.1 | 1.07 | 1.12 | contended |
| Zipf 0.99 | 0.3% | Ours | 45.4 | 0.917 | 0.934 | 7.24 | 4.22 | 0.24 | 7.64 | 11.8 | 12 | 1.71 | 1.83 | contended |
| Zipf 0.99 | 1% | Ours | 47.6 | 0.916 | 0.933 | 7.24 | 4.09 | 0.324 | 7.65 | 11.9 | 12 | 1.77 | 1.91 | contended |
| Zipf 0.99 | 3% | Ours | 52.4 | 0.914 | 0.932 | 7.24 | 3.9 | 0.624 | 7.72 | 11.9 | 12 | 1.86 | 2.1 | contended |
| Zipf 0.99 | 10% | Ours | 55.7 | 0.916 | 0.93 | 7.24 | 3.78 | 0.633 | 7.79 | 12 | 12 | 1.91 | 2.26 | contended |
| Zipf 0.99 | 0.3% | Ours (drift handling) | 45.4 | 0.915 | 0.933 | 7.24 | 4.23 | 0.263 | 7.66 | 12 | 12 | 1.71 | 1.69 | contended |
| Zipf 0.99 | 1% | Ours (drift handling) | 46.9 | 0.912 | 0.931 | 7.24 | 4.27 | 0.496 | 7.73 | 12 | 12 | 1.7 | 1.73 | contended |
| Zipf 0.99 | 3% | Ours (drift handling) | 52.7 | 0.911 | 0.93 | 7.24 | 3.91 | 0.651 | 7.75 | 11.9 | 12 | 1.85 | 1.9 | contended |
| Zipf 0.99 | 10% | Ours (drift handling) | 56.6 | 0.907 | 0.925 | 7.24 | 3.83 | 0.831 | 7.84 | 11.9 | 12 | 1.89 | 2.03 | contended |
| Zipf 0.99 | 0.3% | Ours (continuous greedy) | 48.3 | 0.912 | 0.93 | 7.24 | 4.16 | 0.39 | 7.73 | 12 | 12 | 1.74 | 1.77 | contended |
| Zipf 0.99 | 1% | Ours (continuous greedy) | 51.7 | 0.912 | 0.93 | 7.24 | 3.97 | 0.616 | 7.76 | 11.9 | 12 | 1.82 | 1.87 | contended |
| Zipf 0.99 | 3% | Ours (continuous greedy) | 55.5 | 0.912 | 0.929 | 7.24 | 3.85 | 0.701 | 7.93 | 11.9 | 12 | 1.88 | 2 | contended |
| Zipf 0.99 | 10% | Ours (continuous greedy) | 62.8 | 0.908 | 0.922 | 7.24 | 3.65 | 1.01 | 7.98 | 11.8 | 12 | 1.98 | 2.3 | contended |
| Zipf 0.99 | 0.3% | LRU | 40.1 | 0.917 | 0.937 | 7.24 | 4.76 | 0.462 | 7.75 | 12.2 | 12 | 1.52 | 1.56 | contended |
| Zipf 0.99 | 1% | LRU | 45.8 | 0.915 | 0.934 | 7.24 | 4.54 | 0.764 | 7.88 | 12.2 | 12 | 1.59 | 1.7 | contended |
| Zipf 0.99 | 3% | LRU | 51.6 | 0.913 | 0.932 | 7.24 | 4.29 | 0.87 | 7.95 | 12.1 | 12 | 1.69 | 1.86 | contended |
| Zipf 0.99 | 10% | LRU | 63.2 | 0.91 | 0.924 | 7.24 | 4.08 | 1.56 | 8.4 | 11.7 | 12 | 1.77 | 2.31 | contended |
| Zipf 0.99 | 0.3% | Aker | 25 | 0.897 | 0.937 | 7.24 | 5.48 | 0.0201 | 7.38 | 12 | 12 | 1.32 | 1.33 | contended |
| Zipf 0.99 | 1% | Aker | 34.2 | 0.914 | 0.934 | 7.24 | 4.76 | 0.0157 | 7.38 | 11.8 | 12 | 1.52 | 1.52 | contended |
| Zipf 0.99 | 3% | Aker | 42.9 | 0.918 | 0.929 | 7.24 | 4.11 | 0.0176 | 7.41 | 11.7 | 12 | 1.76 | 1.75 | contended |
| Zipf 0.99 | 10% | Aker | 44.9 | 0.912 | 0.924 | 7.24 | 4.02 | 0.0178 | 7.43 | 11.7 | 12 | 1.8 | 1.81 | contended |
| Zipf 0.99 | 1% | Aker (paper default) | 31.7 | 0.939 | 0.943 | 7.24 | 4.96 | 0.0201 | 7.42 | 11.9 | 12 | 1.46 | 1.46 | contended |
| Zipf 0.99 | 10% | Aker (paper default) | 42.3 | 0.946 | 0.94 | 7.24 | 4.16 | 0.0164 | 7.4 | 11.7 | 12 | 1.74 | 1.73 | contended |
| Zipf 0.99 | 1% | QVCache (paper's D) | 40 | 0.808 | 0.892 | 7.24 | 4.97 | 0.654 | 8.02 | 12.3 | 12 | 1.46 | 1.67 | contended |
| Zipf 0.99 | 10% | QVCache (paper's D) | 38.1 | 0.988 | 0.964 | 7.24 | 4.81 | 0.339 | 7.69 | 12.2 | 12 | 1.51 | 1.62 | contended |
| Zipf 0.99 | 0.3% | QVCache | 31.4 | 0.882 | 0.928 | 7.24 | 5.65 | 0.671 | 8.05 | 12.6 | 12 | 1.28 | 1.46 | contended |
| Zipf 0.99 | 1% | QVCache | 40.8 | 0.78 | 0.879 | 7.24 | 5.29 | 1.04 | 8.41 | 12.7 | 12 | 1.37 | 1.69 | contended |
| Zipf 0.99 | 3% | QVCache | 41.2 | 0.979 | 0.961 | 7.24 | 4.86 | 0.637 | 8 | 12.5 | 12 | 1.49 | 1.7 | contended |
| Zipf 0.99 | 10% | QVCache | 41.2 | 0.978 | 0.96 | 7.24 | 4.41 | 0.173 | 7.56 | 11.8 | 12 | 1.64 | 1.7 | contended |
| ESCI (no repeated queries) | 0.3% | Ours | 6.62 | 0.902 | 0.95 | 6.73 | 6.57 | 0.258 | 6.95 | 10.1 | 9.87 | 1.03 | 1.07 | contended |
| ESCI (no repeated queries) | 1% | Ours | 9.18 | 0.911 | 0.95 | 6.73 | 6.45 | 0.308 | 7.01 | 10.2 | 9.87 | 1.04 | 1.1 | contended |
| ESCI (no repeated queries) | 3% | Ours | 14.5 | 0.899 | 0.945 | 6.73 | 6.32 | 0.451 | 7.17 | 10.3 | 9.87 | 1.06 | 1.17 | contended |
| ESCI (no repeated queries) | 10% | Ours | 22.7 | 0.9 | 0.941 | 6.73 | 5.79 | 0.491 | 7.21 | 10.2 | 9.87 | 1.16 | 1.29 | contended |
| ESCI (no repeated queries) | 0.3% | Ours (drift handling) (cutoff lost 7k requests) | 5.27 | 0.921 | 0.952 | 6.73 | 6.68 | 0.275 | 6.97 | 10.1 | 9.87 | 1.01 | 1.05 | contended |
| ESCI (no repeated queries) | 1% | Ours (drift handling) (cutoff lost 7k requests) (cutoff lost 2k requests) | 8.01 | 0.923 | 0.951 | 6.73 | 6.56 | 0.361 | 7.03 | 10.2 | 9.87 | 1.03 | 1.08 | contended |
| ESCI (no repeated queries) | 3% | Ours (drift handling) (cutoff lost 7k requests) (cutoff lost 2k requests) (cutoff lost 1k requests) | 12.2 | 0.917 | 0.949 | 6.73 | 6.45 | 0.47 | 7.18 | 10.3 | 9.87 | 1.04 | 1.12 | contended |
| ESCI (no repeated queries) | 10% | Ours (drift handling) (cutoff lost 7k requests) (cutoff lost 2k requests) (cutoff lost 1k requests) | 20.3 | 0.913 | 0.945 | 6.73 | 6.06 | 0.577 | 7.26 | 10.2 | 9.87 | 1.11 | 1.22 | contended |
| ESCI (no repeated queries) | 0.3% | Ours (continuous greedy) (cutoff lost 5k requests) | 5.25 | 0.937 | 0.953 | 6.73 | 6.74 | 0.537 | 7.03 | 10.2 | 9.87 | 0.999 | 1.05 | contended |
| ESCI (no repeated queries) | 1% | Ours (continuous greedy) (cutoff lost 5k requests) | 8.98 | 0.927 | 0.951 | 6.73 | 6.64 | 0.685 | 7.12 | 10.3 | 9.87 | 1.01 | 1.09 | contended |
| ESCI (no repeated queries) | 3% | Ours (continuous greedy) (cutoff lost 5k requests) | 13.7 | 0.917 | 0.949 | 6.73 | 6.36 | 0.668 | 7.15 | 10.2 | 9.87 | 1.06 | 1.14 | contended |
| ESCI (no repeated queries) | 10% | Ours (continuous greedy) (cutoff lost 5k requests) | 23.3 | 0.915 | 0.945 | 6.73 | 6.13 | 0.896 | 7.54 | 10.4 | 9.87 | 1.1 | 1.26 | contended |
| ESCI (no repeated queries) | 0.3% | LRU (cutoff lost 103k requests) | 0.0633 | 0.649 | 0.955 | 6.73 | 7.67 | 0.375 | 7.68 | 21.3 | 9.87 | 0.877 | 1 | contended |
| ESCI (no repeated queries) | 1% | LRU (cutoff lost 103k requests) (cutoff lost 85k requests) | 0.74 | 0.92 | 0.954 | 6.73 | 7.81 | 0.445 | 7.84 | 23.5 | 9.87 | 0.862 | 1.01 | contended |
| ESCI (no repeated queries) | 3% | LRU (cutoff lost 103k requests) (cutoff lost 85k requests) (cutoff lost 2k requests) | 8.06 | 0.918 | 0.951 | 6.73 | 8 | 1.55 | 8.51 | 29.9 | 9.87 | 0.841 | 1.08 | contended |
| ESCI (no repeated queries) | 10% | LRU (cutoff lost 103k requests) (cutoff lost 85k requests) (cutoff lost 2k requests) | 15.9 | 0.914 | 0.947 | 6.73 | 8.86 | 3.2 | 9.84 | 31.4 | 9.87 | 0.76 | 1.17 | contended |
| ESCI (no repeated queries) | 0.3% | Aker | 0.00904 | 0.91 | 0.955 | 6.73 | 6.77 | 0.0328 | 6.77 | 10 | 9.87 | 0.995 | 1 | contended |
| ESCI (no repeated queries) | 1% | Aker | 0.0154 | 0.941 | 0.955 | 6.73 | 6.77 | 0.0296 | 6.77 | 10 | 9.87 | 0.995 | 1 | contended |
| ESCI (no repeated queries) | 3% | Aker | 0.0244 | 0.922 | 0.955 | 6.73 | 6.78 | 0.0547 | 6.78 | 10 | 9.87 | 0.993 | 1 | contended |
| ESCI (no repeated queries) | 10% | Aker | 0.112 | 0.934 | 0.955 | 6.73 | 6.79 | 0.0597 | 6.8 | 9.96 | 9.87 | 0.992 | 1 | contended |
| ESCI (no repeated queries) | 1% | Aker (paper default) | 0.103 | 0.74 | 0.955 | 6.73 | 6.77 | 0.0467 | 6.78 | 10 | 9.87 | 0.995 | 1 | contended |
| ESCI (no repeated queries) | 10% | Aker (paper default) | 0.958 | 0.711 | 0.952 | 6.73 | 6.73 | 0.0707 | 6.81 | 9.98 | 9.87 | 1 | 1.01 | contended |
| ESCI (no repeated queries) | 1% | QVCache (paper's D) | 1.76 | 0.293 | 0.943 | 6.73 | 7.24 | 0.538 | 7.36 | 10.5 | 9.87 | 0.929 | 1.02 | contended |
| ESCI (no repeated queries) | 10% | QVCache (paper's D) | 1.76 | 0.716 | 0.951 | 6.73 | 7.84 | 1.06 | 7.96 | 11.2 | 9.87 | 0.859 | 1.02 | contended |
| ESCI (no repeated queries) | 0.3% | QVCache | 0.175 | 0.473 | 0.954 | 6.73 | 7.19 | 0.425 | 7.2 | 10.3 | 9.87 | 0.937 | 1 | contended |
| ESCI (no repeated queries) | 1% | QVCache | 0.268 | 0.628 | 0.954 | 6.73 | 7.02 | 0.215 | 7.04 | 10.2 | 9.87 | 0.959 | 1 | contended |
| ESCI (no repeated queries) | 3% | QVCache | 0.406 | 0.773 | 0.954 | 6.73 | 7.15 | 0.359 | 7.17 | 10.3 | 9.87 | 0.941 | 1 | contended |
| ESCI (no repeated queries) | 10% | QVCache | 0.691 | 0.894 | 0.954 | 6.73 | 8.37 | 1.4 | 8.41 | 11.9 | 9.87 | 0.804 | 1.01 | contended |
| TriviaQA, Zipf 0.3 | 1% | Ours | 7.48 | 0.917 | 0.948 | 73.7 | 68.9 | 3.46 | 74.9 | 82.9 | 81.6 | 1.07 | 1.08 | contended |
| TriviaQA, Zipf 0.3 | 10% | Ours | 30.9 | 0.904 | 0.933 | 73.7 | 56.3 | 14.2 | 76.4 | 83.8 | 81.6 | 1.31 | 1.45 | contended |
| TriviaQA, Zipf 0.3 | 1% | Ours (drift handling) (cutoff lost 3k requests) | 6.09 | 0.926 | 0.95 | 73.7 | 70.3 | 3.94 | 75.3 | 83.4 | 81.6 | 1.05 | 1.06 | contended |
| TriviaQA, Zipf 0.3 | 10% | Ours (drift handling) (cutoff lost 3k requests) (cutoff lost 2k requests) | 38.1 | 0.912 | 0.93 | 73.7 | 51.8 | 14.4 | 77.1 | 84.6 | 81.6 | 1.42 | 1.52 | contended |
| TriviaQA, Zipf 0.3 | 1% | Ours (continuous greedy) | 10.1 | 0.918 | 0.947 | 73.7 | 67.2 | 3.11 | 74.9 | 82.7 | 81.6 | 1.1 | 1.1 | contended |
| TriviaQA, Zipf 0.3 | 10% | Ours (continuous greedy) | 47.3 | 0.914 | 0.925 | 73.7 | 46.3 | 13.6 | 76.8 | 83.8 | 81.6 | 1.59 | 1.74 | contended |
| TriviaQA, Zipf 0.3 | 1% | LRU (cutoff lost 11k requests) | 3.48 | 0.923 | 0.952 | 73.7 | 72.8 | 4.3 | 75.7 | 83.6 | 81.6 | 1.01 | 1.03 | contended |
| TriviaQA, Zipf 0.3 | 10% | LRU (cutoff lost 11k requests) | 54.5 | 0.91 | 0.918 | 73.7 | 44.5 | 16.6 | 79.1 | 85.9 | 81.6 | 1.66 | 1.96 | contended |
| TriviaQA, Zipf 0.3 | 1% | Aker | 2.8 | 0.931 | 0.952 | 73.7 | 71.7 | 0.0422 | 73.8 | 81.7 | 81.6 | 1.03 | 1.03 | contended |
| TriviaQA, Zipf 0.3 | 10% | Aker | 15.3 | 0.958 | 0.948 | 73.7 | 62 | 0.035 | 73.8 | 81.6 | 81.6 | 1.19 | 1.18 | contended |
| TriviaQA, Zipf 0.3 | 1% | QVCache | 2.67 | 0.858 | 0.951 | 73.7 | 76.8 | 4.62 | 78.6 | 87.1 | 81.6 | 0.96 | 1.03 | contended |
| TriviaQA, Zipf 0.3 | 10% | QVCache | 15.6 | 0.909 | 0.943 | 73.7 | 63.7 | 0.897 | 74.7 | 82.7 | 81.6 | 1.16 | 1.18 | contended |
| TriviaQA, Zipf 0.6 | 1% | Ours | 17.2 | 0.916 | 0.942 | 73.5 | 62.7 | 3.68 | 74.8 | 82.8 | 81.9 | 1.17 | 1.21 | contended |
| TriviaQA, Zipf 0.6 | 10% | Ours | 41.8 | 0.907 | 0.928 | 73.5 | 50.4 | 14 | 76 | 84.1 | 81.9 | 1.46 | 1.72 | contended |
| TriviaQA, Zipf 0.6 | 1% | Ours (drift handling) | 17 | 0.916 | 0.942 | 73.5 | 63 | 3.66 | 75 | 83.1 | 81.9 | 1.17 | 1.18 | contended |
| TriviaQA, Zipf 0.6 | 10% | Ours (drift handling) | 41.4 | 0.907 | 0.928 | 73.5 | 50.8 | 14.2 | 76.1 | 84.1 | 81.9 | 1.45 | 1.59 | contended |
| TriviaQA, Zipf 0.6 | 1% | Ours (continuous greedy) | 21.1 | 0.917 | 0.94 | 73.5 | 59.5 | 3.43 | 74.8 | 82.6 | 81.9 | 1.24 | 1.23 | contended |
| TriviaQA, Zipf 0.6 | 10% | Ours (continuous greedy) | 58.8 | 0.909 | 0.917 | 73.5 | 40.8 | 15.8 | 77 | 84.1 | 81.9 | 1.8 | 2.12 | contended |
| TriviaQA, Zipf 0.6 | 1% | LRU | 14.9 | 0.913 | 0.943 | 73.5 | 63.7 | 4.52 | 75.6 | 83.8 | 81.9 | 1.15 | 1.15 | contended |
| TriviaQA, Zipf 0.6 | 10% | LRU | 62 | 0.909 | 0.915 | 73.5 | 39.7 | 17.4 | 78.8 | 85.6 | 81.9 | 1.85 | 2.26 | contended |
| TriviaQA, Zipf 0.6 | 1% | Aker | 6.53 | 0.944 | 0.949 | 73.5 | 68.5 | 0.038 | 73.6 | 81.9 | 81.9 | 1.07 | 1.07 | contended |
| TriviaQA, Zipf 0.6 | 10% | Aker | 27.1 | 0.915 | 0.932 | 73.5 | 54.4 | 0.0403 | 73.6 | 81.4 | 81.9 | 1.35 | 1.37 | contended |
| TriviaQA, Zipf 0.6 | 1% | QVCache | 6.36 | 0.911 | 0.949 | 73.5 | 74.1 | 4.79 | 78.4 | 87.6 | 81.9 | 0.991 | 1.07 | contended |
| TriviaQA, Zipf 0.6 | 10% | QVCache | 24.6 | 0.921 | 0.94 | 73.5 | 56.8 | 0.911 | 74.4 | 82.5 | 81.9 | 1.29 | 1.33 | contended |
| TriviaQA, Zipf 0.99 | 1% | Ours | 61.3 | 0.905 | 0.917 | 74.3 | 30.2 | 3.3 | 75.9 | 82.7 | 84.6 | 2.47 | 2.59 | contended |
| TriviaQA, Zipf 0.99 | 10% | Ours | 74.7 | 0.908 | 0.916 | 74.3 | 28.6 | 13.5 | 76.3 | 82.6 | 84.6 | 2.6 | 3.95 | contended |
| TriviaQA, Zipf 0.99 | 1% | Ours (drift handling) | 60.2 | 0.913 | 0.923 | 74.3 | 31.2 | 3.56 | 76.1 | 82.9 | 84.6 | 2.38 | 2.18 | contended |
| TriviaQA, Zipf 0.99 | 10% | Ours (drift handling) | 74.6 | 0.908 | 0.917 | 74.3 | 28.6 | 13.7 | 76.4 | 82.6 | 84.6 | 2.6 | 3.03 | contended |
| TriviaQA, Zipf 0.99 | 1% | Ours (continuous greedy) | 63.4 | 0.908 | 0.918 | 74.3 | 29 | 3.62 | 75.9 | 82.3 | 84.6 | 2.56 | 2.32 | contended |
| TriviaQA, Zipf 0.99 | 10% | Ours (continuous greedy) | 86.2 | 0.908 | 0.907 | 74.3 | 23.2 | 14.6 | 76.8 | 81.5 | 84.6 | 3.2 | 4.43 | contended |
| TriviaQA, Zipf 0.99 | 1% | LRU | 58.1 | 0.906 | 0.92 | 74.3 | 33.3 | 4.17 | 76.3 | 83.2 | 84.6 | 2.23 | 2.09 | contended |
| TriviaQA, Zipf 0.99 | 10% | LRU | 86.9 | 0.909 | 0.908 | 74.3 | 23.3 | 15.4 | 78.3 | 82.4 | 84.6 | 3.19 | 4.56 | contended |
| TriviaQA, Zipf 0.99 | 1% | Aker | 43.2 | 0.93 | 0.925 | 74.3 | 41.8 | 0.0302 | 74.5 | 82.8 | 84.6 | 1.78 | 1.76 | contended |
| TriviaQA, Zipf 0.99 | 10% | Aker | 59.2 | 0.937 | 0.918 | 74.3 | 29.1 | 0.0276 | 74.5 | 81.6 | 84.6 | 2.55 | 2.45 | contended |
| TriviaQA, Zipf 0.99 | 1% | QVCache | 59.4 | 0.903 | 0.918 | 74.3 | 33.1 | 4.46 | 78.9 | 86.6 | 84.6 | 2.25 | 2.46 | contended |
| TriviaQA, Zipf 0.99 | 10% | QVCache | 59.7 | 0.945 | 0.939 | 74.3 | 29.2 | 0.821 | 75.2 | 82.3 | 84.6 | 2.55 | 2.48 | contended |
| SPACEV real query log | 1% | Ours (drift handling) (cutoff lost 50k requests) | 2.6 | 0.919 | 0.913 | 60.6 | 59.8 | 1.76 | 61.1 | 73.1 | 73.6 | 1.01 | 1.02 | contended |
| SPACEV real query log | 10% | Ours (drift handling) (cutoff lost 50k requests) (cutoff lost 32k requests) | 3.57 | 0.921 | 0.913 | 60.6 | 61.1 | 6.49 | 63.2 | 75.4 | 73.6 | 0.992 | 1.03 | contended |
| SPACEV real query log | 1% | Ours (continuous greedy) (cutoff lost 19k requests) | 3.96 | 0.922 | 0.913 | 60.6 | 58.8 | 1.81 | 61 | 72.7 | 73.6 | 1.03 | 1.04 | contended |
| SPACEV real query log | 10% | Ours (continuous greedy) (cutoff lost 19k requests) (cutoff lost 12k requests) | 5.32 | 0.923 | 0.914 | 60.6 | 60.5 | 7.08 | 63.3 | 75.1 | 73.6 | 1 | 1.05 | contended |
| SPACEV real query log | 1% | LRU (cutoff lost 27k requests) | 2.61 | 0.926 | 0.913 | 60.6 | 61.7 | 5.12 | 63.4 | 103 | 73.6 | 0.982 | 1.02 | contended |
| SPACEV real query log | 10% | LRU (cutoff lost 27k requests) (cutoff lost 5k requests) | 6.92 | 0.924 | 0.914 | 60.6 | 63.3 | 12.2 | 67.1 | 97.7 | 73.6 | 0.957 | 1.07 | contended |
| SPACEV real query log | 1% | Aker | 3.25 | 0.944 | 0.911 | 60.6 | 58.3 | 0.0218 | 60.6 | 72.7 | 73.6 | 1.04 | 1.03 | contended |
| SPACEV real query log | 10% | Aker | 8.25 | 0.915 | 0.907 | 60.6 | 55.7 | 0.0227 | 60.6 | 72.3 | 73.6 | 1.09 | 1.09 | contended |
| SPACEV real query log | 1% | QVCache | 2.39 | 0.959 | 0.913 | 60.6 | 60.5 | 1.51 | 62.1 | 75.2 | 73.6 | 1 | 1.02 | contended |
| SPACEV real query log | 10% | QVCache | 7.54 | 0.738 | 0.897 | 60.6 | 56.6 | 0.499 | 61.1 | 73.3 | 73.6 | 1.07 | 1.08 | contended |
| TripClick real health-search log | 0.3% | Ours (drift handling) (cutoff lost 28k requests) | 1.49 | 0.928 | 0.982 | 9.42 | 9.6 | 0.397 | 9.75 | 14 | 13.6 | 0.981 | 1.01 | contended |
| TripClick real health-search log | 1% | Ours (drift handling) (cutoff lost 28k requests) (cutoff lost 21k requests) | 2.7 | 0.928 | 0.981 | 9.42 | 9.72 | 0.658 | 9.97 | 14.2 | 13.6 | 0.969 | 1.02 | contended |
| TripClick real health-search log | 3% | Ours (drift handling) (cutoff lost 28k requests) (cutoff lost 21k requests) (cutoff lost 22k requests) | 5.91 | 0.92 | 0.979 | 9.42 | 9.87 | 1.36 | 10.5 | 14.4 | 13.6 | 0.954 | 1.06 | contended |
| TripClick real health-search log | 10% | Ours (drift handling) (cutoff lost 28k requests) (cutoff lost 21k requests) (cutoff lost 22k requests) (cutoff lost 24k requests) | 10.9 | 0.911 | 0.974 | 9.42 | 10 | 2.06 | 11 | 14.9 | 13.6 | 0.942 | 1.11 | contended |
| TripClick real health-search log | 0.3% | Ours (continuous greedy) (cutoff lost 10k requests) | 3.88 | 0.942 | 0.981 | 9.42 | 9.45 | 0.46 | 9.79 | 14.1 | 13.6 | 0.997 | 1.04 | contended |
| TripClick real health-search log | 1% | Ours (continuous greedy) (cutoff lost 10k requests) (cutoff lost 4k requests) | 6.06 | 0.925 | 0.979 | 9.42 | 9.39 | 0.622 | 9.93 | 14.2 | 13.6 | 1 | 1.06 | contended |
| TripClick real health-search log | 3% | Ours (continuous greedy) (cutoff lost 10k requests) (cutoff lost 4k requests) (cutoff lost 1k requests) | 11.2 | 0.918 | 0.975 | 9.42 | 9.11 | 1.07 | 10.1 | 14.2 | 13.6 | 1.03 | 1.11 | contended |
| TripClick real health-search log | 10% | Ours (continuous greedy) (cutoff lost 10k requests) (cutoff lost 4k requests) (cutoff lost 1k requests) | 25.2 | 0.911 | 0.962 | 9.42 | 8.48 | 1.76 | 10.6 | 14.7 | 13.6 | 1.11 | 1.29 | contended |
| TripClick real health-search log | 0.3% | LRU | 13.7 | 0.926 | 0.974 | 9.42 | 9.37 | 1.16 | 10.6 | 25.5 | 13.6 | 1.01 | 1.14 | contended |
| TripClick real health-search log | 1% | LRU | 16.6 | 0.925 | 0.972 | 9.42 | 9.79 | 1.71 | 11.4 | 33.2 | 13.6 | 0.962 | 1.17 | contended |
| TripClick real health-search log | 3% | LRU | 22.7 | 0.917 | 0.966 | 9.42 | 9.76 | 2.85 | 11.8 | 31.6 | 13.6 | 0.965 | 1.26 | contended |
| TripClick real health-search log | 10% | LRU | 32.6 | 0.913 | 0.957 | 9.42 | 10.8 | 4.82 | 13.5 | 34.1 | 13.6 | 0.873 | 1.41 | contended |
| TripClick real health-search log | 0.3% | Aker | 9.22 | 0.914 | 0.975 | 9.42 | 8.62 | 0.0292 | 9.47 | 13.5 | 13.6 | 1.09 | 1.1 | contended |
| TripClick real health-search log | 1% | Aker | 11.1 | 0.925 | 0.975 | 9.42 | 8.54 | 0.0331 | 9.5 | 13.5 | 13.6 | 1.1 | 1.12 | contended |
| TripClick real health-search log | 3% | Aker | 13.7 | 0.927 | 0.973 | 9.42 | 8.32 | 0.0359 | 9.51 | 13.5 | 13.6 | 1.13 | 1.16 | contended |
| TripClick real health-search log | 10% | Aker | 18.3 | 0.937 | 0.972 | 9.42 | 7.84 | 0.0446 | 9.56 | 13.5 | 13.6 | 1.2 | 1.22 | contended |
| TripClick real health-search log | 0.3% | QVCache | 7.47 | 0.826 | 0.971 | 9.42 | 9.65 | 0.843 | 10.3 | 14.5 | 13.6 | 0.976 | 1.08 | contended |
| TripClick real health-search log | 1% | QVCache | 9.48 | 0.812 | 0.967 | 9.42 | 11.5 | 2.79 | 12.3 | 16.7 | 13.6 | 0.821 | 1.1 | contended |
| TripClick real health-search log | 3% | QVCache | 11.9 | 0.824 | 0.964 | 9.42 | 10.3 | 1.86 | 11.4 | 15.5 | 13.6 | 0.913 | 1.13 | contended |
| TripClick real health-search log | 10% | QVCache | 16.4 | 0.896 | 0.968 | 9.42 | 12.7 | 4.69 | 14.2 | 18.6 | 13.6 | 0.741 | 1.2 | contended |
| MedRAG-Zipf (Proximity) | 0.01% | Ours | 42.5 | 0.92 | 0.952 | 11.5 | 6.74 | 0.276 | 11.9 | 15 | 15.1 | 1.7 | 1.74 | contended |
| MedRAG-Zipf (Proximity) | 0.03% | Ours | 54.5 | 0.921 | 0.945 | 11.5 | 5.34 | 0.434 | 12.1 | 14.9 | 15.1 | 2.15 | 2.2 | contended |
| MedRAG-Zipf (Proximity) | 0.1% | Ours | 64.2 | 0.916 | 0.939 | 11.5 | 5.28 | 1.35 | 12.8 | 15.6 | 15.1 | 2.18 | 2.79 | contended |
| MedRAG-Zipf (Proximity) | 0.3% | Ours | 64.7 | 0.922 | 0.942 | 11.5 | 5.74 | 1.69 | 13.1 | 16.1 | 15.1 | 2 | 2.83 | contended |
| MedRAG-Zipf (Proximity) | 0.01% | Ours (drift handling) | 42.5 | 0.919 | 0.951 | 11.5 | 6.76 | 0.302 | 11.9 | 15 | 15.1 | 1.7 | 1.62 | contended |
| MedRAG-Zipf (Proximity) | 0.03% | Ours (drift handling) | 55.5 | 0.918 | 0.943 | 11.5 | 5.18 | 0.508 | 12.2 | 14.9 | 15.1 | 2.22 | 1.99 | contended |
| MedRAG-Zipf (Proximity) | 0.1% | Ours (drift handling) | 63.5 | 0.924 | 0.945 | 11.5 | 5.21 | 1.27 | 12.7 | 15.5 | 15.1 | 2.21 | 2.31 | contended |
| MedRAG-Zipf (Proximity) | 0.3% | Ours (drift handling) | 65.7 | 0.91 | 0.933 | 11.5 | 5.55 | 1.72 | 13.1 | 15.8 | 15.1 | 2.07 | 2.43 | contended |
| MedRAG-Zipf (Proximity) | 0.01% | Ours (continuous greedy) | 49.1 | 0.921 | 0.948 | 11.5 | 6.43 | 0.43 | 12 | 15.1 | 15.1 | 1.79 | 1.79 | contended |
| MedRAG-Zipf (Proximity) | 0.03% | Ours (continuous greedy) | 66 | 0.912 | 0.932 | 11.5 | 4.13 | 0.591 | 11.9 | 14.5 | 15.1 | 2.78 | 2.45 | contended |
| MedRAG-Zipf (Proximity) | 0.1% | Ours (continuous greedy) | 77.9 | 0.918 | 0.931 | 11.5 | 3.65 | 1.29 | 12.4 | 14.7 | 15.1 | 3.15 | 3.3 | contended |
| MedRAG-Zipf (Proximity) | 0.3% | Ours (continuous greedy) | 88 | 0.915 | 0.923 | 11.5 | 3.94 | 2.58 | 13 | 15.9 | 15.1 | 2.91 | 4.68 | contended |
| MedRAG-Zipf (Proximity) | 0.01% | LRU | 34.2 | 0.921 | 0.958 | 11.5 | 8.11 | 0.613 | 12 | 15.5 | 15.1 | 1.42 | 1.44 | contended |
| MedRAG-Zipf (Proximity) | 0.03% | LRU | 57.4 | 0.916 | 0.94 | 11.5 | 5.64 | 0.753 | 12.3 | 15.2 | 15.1 | 2.04 | 2.05 | contended |
| MedRAG-Zipf (Proximity) | 0.1% | LRU | 77.3 | 0.896 | 0.914 | 11.5 | 4.23 | 1.55 | 12.8 | 15.8 | 15.1 | 2.72 | 3.25 | contended |
| MedRAG-Zipf (Proximity) | 0.3% | LRU | 92.7 | 0.911 | 0.916 | 11.5 | 2.88 | 1.99 | 12.9 | 14.4 | 15.1 | 3.99 | 5.88 | contended |
| MedRAG-Zipf (Proximity) | 0.01% | Aker | 1.76 | 0.899 | 0.98 | 11.5 | 11.5 | 0.0364 | 11.5 | 15.2 | 15.1 | 1 | 1.02 | contended |
| MedRAG-Zipf (Proximity) | 0.03% | Aker | 4.9 | 0.904 | 0.977 | 11.5 | 11.3 | 0.0451 | 11.6 | 15.2 | 15.1 | 1.02 | 1.05 | contended |
| MedRAG-Zipf (Proximity) | 0.1% | Aker | 16.7 | 0.904 | 0.967 | 11.5 | 9.77 | 0.0498 | 11.6 | 15.2 | 15.1 | 1.18 | 1.2 | contended |
| MedRAG-Zipf (Proximity) | 0.3% | Aker | 16.7 | 0.904 | 0.967 | 11.5 | 9.77 | 0.047 | 11.6 | 15.2 | 15.1 | 1.18 | 1.2 | contended |
| MedRAG-Zipf (Proximity) | 0.01% | QVCache | 18.3 | 0.762 | 0.939 | 11.5 | 10.1 | 0.318 | 11.9 | 15.5 | 15.1 | 1.14 | 1.22 | contended |
| MedRAG-Zipf (Proximity) | 0.03% | QVCache | 29.6 | 0.907 | 0.957 | 11.5 | 9.63 | 1.59 | 13.1 | 15.7 | 15.1 | 1.19 | 1.42 | contended |
| MedRAG-Zipf (Proximity) | 0.1% | QVCache | 46.4 | 0.956 | 0.968 | 11.5 | 6.88 | 0.526 | 11.9 | 15.2 | 15.1 | 1.67 | 1.86 | contended |
| MedRAG-Zipf (Proximity) | 0.3% | QVCache | 46.4 | 0.956 | 0.968 | 11.5 | 6.72 | 0.37 | 11.8 | 15 | 15.1 | 1.71 | 1.86 | contended |

Sample-rate check (mean ms, cache + DiskANN / DiskANN alone / speedup; whole-run hit rate)

- Zipf 0.6, 1%, Ours: 1/20: pending; 1/10: pending; 1/5: pending
- Zipf 0.6, 1%, LRU: 1/20: pending; 1/10: pending; 1/5: pending
- Zipf 0.99, 10%, Ours: 1/20: pending; 1/10: pending; 1/5: pending
- Zipf 0.99, 10%, LRU: 1/20: pending; 1/10: pending; 1/5: pending
