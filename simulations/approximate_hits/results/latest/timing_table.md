| trace | budget | technique | hit % | ret. hits | recall all | DiskANN ms | cache+DiskANN ms | hit ms | miss ms | p99 ms | DiskANN p99 | speedup | 1/(1-h) | env |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zipf 0.3 | 0.3% | Ours | 1.06 | 0.944 | 0.956 | 10.2 | 10.8 | 0.592 | 10.8 | 16.7 | 16 | 0.942 | 1.01 | contended |
| Zipf 0.3 | 1% | Ours (cutoff lost 79k requests) | 0 | - | 0.956 | 10.2 | 11.5 | - | 11.5 | 17.5 | 16 | 0.882 | 1 | contended |
| Zipf 0.3 | 3% | Ours (cutoff lost 79k requests) (cutoff lost 79k requests) | 0 | - | 0.956 | 10.2 | 13.3 | - | 13.3 | 19.1 | 16 | 0.765 | 1 | contended |
| Zipf 0.3 | 10% | Ours (cutoff lost 79k requests) (cutoff lost 79k requests) | 12.1 | 0.922 | 0.95 | 10.2 | 11 | 1.92 | 12.4 | 18.3 | 16 | 0.924 | 1.14 | contended |
| Zipf 0.3 | 0.3% | Ours (drift handling) (cutoff lost 37k requests) | 0.988 | 0.919 | 0.956 | 10.2 | 10.6 | 0.453 | 10.7 | 16.5 | 16 | 0.956 | 1.01 | contended |
| Zipf 0.3 | 1% | Ours (drift handling) (cutoff lost 37k requests) (cutoff lost 27k requests) | 2.28 | 0.92 | 0.955 | 10.2 | 11.2 | 2.39 | 11.4 | 17.3 | 16 | 0.906 | 1.02 | contended |
| Zipf 0.3 | 3% | Ours (drift handling) (cutoff lost 37k requests) (cutoff lost 27k requests) (cutoff lost 35k requests) | 2.57 | 0.941 | 0.955 | 10.2 | 11.5 | 2.89 | 11.8 | 17.6 | 16 | 0.881 | 1.02 | contended |
| Zipf 0.3 | 10% | Ours (drift handling) (cutoff lost 37k requests) (cutoff lost 27k requests) (cutoff lost 35k requests) | 11.8 | 0.923 | 0.951 | 10.2 | 11.1 | 2.77 | 12.4 | 18.1 | 16 | 0.918 | 1.12 | contended |
| Zipf 0.3 | 0.3% | Ours (continuous greedy) (cutoff lost 39k requests) | 0.847 | 0.913 | 0.956 | 10.2 | 10.6 | 0.546 | 10.7 | 16.6 | 16 | 0.955 | 1.01 | contended |
| Zipf 0.3 | 1% | Ours (continuous greedy) (cutoff lost 39k requests) (cutoff lost 29k requests) | 1.94 | 0.931 | 0.955 | 10.2 | 10.8 | 2.55 | 11 | 16.9 | 16 | 0.944 | 1.02 | contended |
| Zipf 0.3 | 3% | Ours (continuous greedy) (cutoff lost 39k requests) (cutoff lost 29k requests) (cutoff lost 2k requests) | 6.55 | 0.932 | 0.953 | 10.2 | 11.2 | 2.23 | 12 | 18 | 16 | 0.903 | 1.06 | contended |
| Zipf 0.3 | 10% | Ours (continuous greedy) (cutoff lost 39k requests) (cutoff lost 29k requests) (cutoff lost 2k requests) | 17 | 0.919 | 0.947 | 10.2 | 9.78 | 2.19 | 11.6 | 17.4 | 16 | 1.04 | 1.18 | contended |
| Zipf 0.3 | 0.3% | LRU (cutoff lost 43k requests) | 0.672 | 0.933 | 0.956 | 10.2 | 11.4 | 1.36 | 11.5 | 17.3 | 16 | 0.888 | 1.01 | contended |
| Zipf 0.3 | 1% | LRU (cutoff lost 43k requests) (cutoff lost 19k requests) | 2.32 | 0.937 | 0.955 | 10.2 | 11.1 | 1.57 | 11.3 | 17.2 | 16 | 0.913 | 1.02 | contended |
| Zipf 0.3 | 3% | LRU (cutoff lost 43k requests) (cutoff lost 19k requests) (cutoff lost 3k requests) | 7.49 | 0.916 | 0.952 | 10.2 | 11.7 | 3.03 | 12.4 | 18.3 | 16 | 0.864 | 1.07 | contended |
| Zipf 0.3 | 10% | LRU (cutoff lost 43k requests) (cutoff lost 19k requests) (cutoff lost 3k requests) | 21.9 | 0.915 | 0.944 | 10.2 | 10.1 | 2.45 | 12.2 | 17.8 | 16 | 1.01 | 1.24 | contended |
| Zipf 0.3 | 0.3% | Aker | 0.29 | 0.921 | 0.956 | 10.2 | 10.2 | 0.0239 | 10.2 | 16.1 | 16 | 0.999 | 1 | contended |
| Zipf 0.3 | 1% | Aker | 0.886 | 0.926 | 0.955 | 10.2 | 10.1 | 0.0338 | 10.2 | 16.1 | 16 | 1 | 1.01 | contended |
| Zipf 0.3 | 3% | Aker | 2.66 | 0.936 | 0.955 | 10.2 | 9.94 | 0.0366 | 10.2 | 16 | 16 | 1.02 | 1.03 | contended |
| Zipf 0.3 | 10% | Aker | 7.05 | 0.899 | 0.949 | 10.2 | 9.51 | 0.0368 | 10.2 | 16 | 16 | 1.07 | 1.08 | contended |
| Zipf 0.3 | 1% | Aker (paper default) | 3.29 | 0.761 | 0.949 | 10.2 | 9.84 | 0.0401 | 10.2 | 16 | 16 | 1.03 | 1.03 | contended |
| Zipf 0.3 | 10% | Aker (paper default) | 9.03 | 0.812 | 0.94 | 10.2 | 9.32 | 0.0356 | 10.2 | 16 | 16 | 1.09 | 1.1 | contended |
| Zipf 0.3 | 1% | QVCache (paper's D) | 5.47 | 0.34 | 0.922 | 10.2 | 10.3 | 0.635 | 10.8 | 16.7 | 16 | 0.99 | 1.06 | contended |
| Zipf 0.3 | 10% | QVCache (paper's D) | 5.5 | 0.954 | 0.955 | 10.2 | 10 | 0.432 | 10.6 | 16.5 | 16 | 1.01 | 1.06 | contended |
| Zipf 0.3 | 0.3% | QVCache | 0.204 | 0.939 | 0.956 | 10.2 | 10.9 | 0.724 | 11 | 16.9 | 16 | 0.928 | 1 | contended |
| Zipf 0.3 | 1% | QVCache | 1.38 | 0.878 | 0.955 | 10.2 | 11.4 | 1.34 | 11.5 | 17.5 | 16 | 0.889 | 1.01 | contended |
| Zipf 0.3 | 3% | QVCache | 3.98 | 0.922 | 0.954 | 10.2 | 11.4 | 1.71 | 11.8 | 17.9 | 16 | 0.889 | 1.04 | contended |
| Zipf 0.3 | 10% | QVCache | 5.55 | 0.946 | 0.954 | 10.2 | 9.88 | 0.245 | 10.4 | 16.3 | 16 | 1.03 | 1.06 | contended |
| Zipf 0.6 | 0.3% | Ours | 8.83 | 0.929 | 0.948 | 9.61 | 9.58 | 0.697 | 10.4 | 14.2 | 13.6 | 1 | 1.1 | contended |
| Zipf 0.6 | 1% | Ours | 10.7 | 0.931 | 0.947 | 9.61 | 9.52 | 0.753 | 10.5 | 14.1 | 13.6 | 1.01 | 1.12 | contended |
| Zipf 0.6 | 3% | Ours | 14.8 | 0.91 | 0.943 | 9.61 | 9.19 | 0.862 | 10.6 | 14.4 | 13.6 | 1.04 | 1.17 | contended |
| Zipf 0.6 | 10% | Ours | 20.5 | 0.923 | 0.942 | 9.61 | 8.99 | 1.09 | 11 | 14.6 | 13.6 | 1.07 | 1.26 | contended |
| Zipf 0.6 | 0.3% | Ours (drift handling) | 9.08 | 0.923 | 0.947 | 9.61 | 9.48 | 1.68 | 10.2 | 14.1 | 13.6 | 1.01 | 1.09 | contended |
| Zipf 0.6 | 1% | Ours (drift handling) | 11 | 0.925 | 0.947 | 9.61 | 9.6 | 1.58 | 10.4 | 14.3 | 13.6 | 1 | 1.11 | contended |
| Zipf 0.6 | 3% | Ours (drift handling) | 13.4 | 0.927 | 0.946 | 9.61 | 10.2 | 2.49 | 11.3 | 15.1 | 13.6 | 0.946 | 1.14 | contended |
| Zipf 0.6 | 10% | Ours (drift handling) | 19.9 | 0.926 | 0.943 | 9.61 | 9.28 | 2.35 | 11 | 14.7 | 13.6 | 1.04 | 1.22 | contended |
| Zipf 0.6 | 0.3% | Ours (continuous greedy) | 11.3 | 0.925 | 0.946 | 9.61 | 9.4 | 1.1 | 10.4 | 14.2 | 13.6 | 1.02 | 1.11 | contended |
| Zipf 0.6 | 1% | Ours (continuous greedy) | 15.1 | 0.921 | 0.944 | 9.61 | 9.28 | 1.63 | 10.6 | 14.4 | 13.6 | 1.04 | 1.16 | contended |
| Zipf 0.6 | 3% | Ours (continuous greedy) | 19.2 | 0.915 | 0.941 | 9.61 | 9.04 | 1.87 | 10.7 | 14.6 | 13.6 | 1.06 | 1.21 | contended |
| Zipf 0.6 | 10% | Ours (continuous greedy) | 27.3 | 0.922 | 0.939 | 9.61 | 8.63 | 2.21 | 11.1 | 14.9 | 13.6 | 1.11 | 1.32 | contended |
| Zipf 0.6 | 0.3% | LRU (cutoff lost 6k requests) | 4.92 | 0.918 | 0.949 | 9.61 | 10.1 | 1.9 | 10.5 | 14.5 | 13.6 | 0.95 | 1.05 | contended |
| Zipf 0.6 | 1% | LRU (cutoff lost 6k requests) | 9.76 | 0.924 | 0.947 | 9.61 | 10 | 1.83 | 10.8 | 14.7 | 13.6 | 0.96 | 1.1 | contended |
| Zipf 0.6 | 3% | LRU (cutoff lost 6k requests) | 15.3 | 0.917 | 0.944 | 9.61 | 9.88 | 2.4 | 11.1 | 15 | 13.6 | 0.973 | 1.16 | contended |
| Zipf 0.6 | 10% | LRU (cutoff lost 6k requests) | 31.6 | 0.913 | 0.935 | 9.61 | 8.9 | 2.88 | 11.7 | 15.2 | 13.6 | 1.08 | 1.4 | contended |
| Zipf 0.6 | 0.3% | Aker | 1.57 | 0.885 | 0.95 | 9.61 | 9.49 | 0.0258 | 9.64 | 13.6 | 13.6 | 1.01 | 1.02 | contended |
| Zipf 0.6 | 1% | Aker | 3.76 | 0.89 | 0.948 | 9.61 | 9.38 | 0.0298 | 9.66 | 13.6 | 13.6 | 1.02 | 1.04 | contended |
| Zipf 0.6 | 3% | Aker | 8.13 | 0.909 | 0.945 | 9.61 | 8.87 | 0.0312 | 9.67 | 13.5 | 13.6 | 1.08 | 1.09 | contended |
| Zipf 0.6 | 10% | Aker | 12.6 | 0.918 | 0.942 | 9.61 | 8.51 | 0.0294 | 9.68 | 13.5 | 13.6 | 1.13 | 1.14 | contended |
| Zipf 0.6 | 1% | Aker (paper default) | 6.24 | 0.809 | 0.941 | 9.61 | 9.03 | 0.0378 | 9.67 | 13.6 | 13.6 | 1.06 | 1.07 | contended |
| Zipf 0.6 | 10% | Aker (paper default) | 13.9 | 0.86 | 0.934 | 9.61 | 8.29 | 0.0283 | 9.67 | 13.4 | 13.6 | 1.16 | 1.16 | contended |
| Zipf 0.6 | 1% | QVCache (paper's D) | 10.6 | 0.497 | 0.902 | 9.61 | 9.22 | 0.621 | 10.2 | 14.1 | 13.6 | 1.04 | 1.12 | contended |
| Zipf 0.6 | 10% | QVCache (paper's D) | 10.1 | 0.971 | 0.951 | 9.61 | 9.03 | 0.409 | 10 | 13.9 | 13.6 | 1.06 | 1.11 | contended |
| Zipf 0.6 | 0.3% | QVCache | 1.27 | 0.953 | 0.951 | 9.61 | 10.3 | 0.833 | 10.4 | 14.4 | 13.6 | 0.934 | 1.01 | contended |
| Zipf 0.6 | 1% | QVCache | 5.01 | 0.876 | 0.947 | 9.61 | 9.54 | 0.42 | 10 | 14 | 13.6 | 1.01 | 1.05 | contended |
| Zipf 0.6 | 3% | QVCache | 10.4 | 0.924 | 0.947 | 9.61 | 10.2 | 1.64 | 11.2 | 15.2 | 13.6 | 0.941 | 1.12 | contended |
| Zipf 0.6 | 10% | QVCache | 10.6 | 0.958 | 0.95 | 9.61 | 8.84 | 0.248 | 9.87 | 13.7 | 13.6 | 1.09 | 1.12 | contended |
| Zipf 0.99 | 0.3% | Ours | 45.4 | 0.92 | 0.936 | 9.53 | 5.98 | 0.665 | 10.5 | 14.9 | 14.3 | 1.59 | 1.83 | contended |
| Zipf 0.99 | 1% | Ours | 48.5 | 0.912 | 0.932 | 9.53 | 6.31 | 1.32 | 11.2 | 15.4 | 14.3 | 1.51 | 1.94 | contended |
| Zipf 0.99 | 3% | Ours | 53.6 | 0.905 | 0.927 | 9.53 | 5.24 | 0.844 | 10.7 | 15 | 14.3 | 1.82 | 2.15 | contended |
| Zipf 0.99 | 10% | Ours | 57 | 0.91 | 0.928 | 9.53 | 5.32 | 1.08 | 11 | 15.2 | 14.3 | 1.79 | 2.33 | contended |
| Zipf 0.99 | 0.3% | Ours (drift handling) | 45.5 | 0.917 | 0.935 | 9.53 | 6.19 | 1.4 | 10.3 | 14.7 | 14.3 | 1.54 | 1.69 | contended |
| Zipf 0.99 | 1% | Ours (drift handling) | 48.9 | 0.91 | 0.931 | 9.53 | 6.06 | 1.67 | 10.5 | 15 | 14.3 | 1.57 | 1.78 | contended |
| Zipf 0.99 | 3% | Ours (drift handling) | 52.3 | 0.917 | 0.934 | 9.53 | 5.84 | 1.72 | 10.7 | 15.1 | 14.3 | 1.63 | 1.88 | contended |
| Zipf 0.99 | 10% | Ours (drift handling) | 56.5 | 0.913 | 0.93 | 9.53 | 5.83 | 1.99 | 11 | 15.6 | 14.3 | 1.63 | 2.03 | contended |
| Zipf 0.99 | 0.3% | Ours (continuous greedy) | 48.3 | 0.917 | 0.934 | 9.53 | 6.09 | 1.49 | 10.5 | 14.8 | 14.3 | 1.56 | 1.77 | contended |
| Zipf 0.99 | 1% | Ours (continuous greedy) | 52 | 0.914 | 0.932 | 9.53 | 6.1 | 1.93 | 10.7 | 15 | 14.3 | 1.56 | 1.87 | contended |
| Zipf 0.99 | 3% | Ours (continuous greedy) | 55.3 | 0.916 | 0.933 | 9.53 | 5.69 | 1.75 | 10.8 | 15.3 | 14.3 | 1.68 | 1.99 | contended |
| Zipf 0.99 | 10% | Ours (continuous greedy) | 63.1 | 0.913 | 0.927 | 9.53 | 5.49 | 2.05 | 11.3 | 15.9 | 14.3 | 1.73 | 2.31 | contended |
| Zipf 0.99 | 0.3% | LRU | 40.8 | 0.909 | 0.934 | 9.53 | 6.9 | 1.77 | 10.6 | 15.2 | 14.3 | 1.38 | 1.58 | contended |
| Zipf 0.99 | 1% | LRU | 46.2 | 0.912 | 0.933 | 9.53 | 6.48 | 1.73 | 10.7 | 15.2 | 14.3 | 1.47 | 1.71 | contended |
| Zipf 0.99 | 3% | LRU | 51.8 | 0.914 | 0.933 | 9.53 | 6.47 | 2.26 | 11.1 | 15.6 | 14.3 | 1.47 | 1.87 | contended |
| Zipf 0.99 | 10% | LRU | 63.1 | 0.916 | 0.929 | 9.53 | 5.89 | 2.54 | 11.5 | 15.8 | 14.3 | 1.62 | 2.31 | contended |
| Zipf 0.99 | 0.3% | Aker | 25 | 0.897 | 0.937 | 9.53 | 7.2 | 0.0222 | 9.7 | 14.4 | 14.3 | 1.32 | 1.33 | contended |
| Zipf 0.99 | 1% | Aker | 34.2 | 0.914 | 0.934 | 9.53 | 6.1 | 0.0159 | 9.46 | 13.7 | 14.3 | 1.56 | 1.52 | quiet |
| Zipf 0.99 | 3% | Aker | 42.9 | 0.918 | 0.929 | 9.53 | 5.41 | 0.0188 | 9.75 | 14.2 | 14.3 | 1.76 | 1.75 | contended |
| Zipf 0.99 | 10% | Aker | 44.9 | 0.912 | 0.924 | 9.53 | 5.21 | 0.0175 | 9.63 | 13.6 | 14.3 | 1.83 | 1.81 | quiet |
| Zipf 0.99 | 1% | Aker (paper default) | 31.7 | 0.939 | 0.943 | 9.53 | 6.51 | 0.0198 | 9.73 | 14.3 | 14.3 | 1.46 | 1.46 | contended |
| Zipf 0.99 | 10% | Aker (paper default) | 42.3 | 0.946 | 0.94 | 9.53 | 5.47 | 0.017 | 9.75 | 14.2 | 14.3 | 1.74 | 1.73 | contended |
| Zipf 0.99 | 1% | QVCache (paper's D) | 40 | 0.808 | 0.892 | 9.53 | 6.29 | 0.603 | 10.3 | 14.8 | 14.3 | 1.52 | 1.67 | contended |
| Zipf 0.99 | 10% | QVCache (paper's D) | 38.1 | 0.988 | 0.964 | 9.53 | 6.21 | 0.318 | 10 | 14.5 | 14.3 | 1.54 | 1.62 | contended |
| Zipf 0.99 | 0.3% | QVCache | 31.4 | 0.882 | 0.928 | 9.53 | 7.4 | 0.85 | 10.6 | 15 | 14.3 | 1.29 | 1.46 | contended |
| Zipf 0.99 | 1% | QVCache | 40.8 | 0.78 | 0.879 | 9.53 | 6.4 | 0.619 | 10.6 | 15.1 | 14.3 | 1.49 | 1.69 | quiet |
| Zipf 0.99 | 3% | QVCache | 41.2 | 0.979 | 0.961 | 9.53 | 6.22 | 0.653 | 10.4 | 14.9 | 14.3 | 1.53 | 1.7 | contended |
| Zipf 0.99 | 10% | QVCache | 41.2 | 0.978 | 0.96 | 9.53 | 5.78 | 0.163 | 9.96 | 14 | 14.3 | 1.65 | 1.7 | quiet |
| ESCI (no repeated queries) | 0.3% | Ours | 6.63 | 0.905 | 0.95 | 8.46 | 8.47 | 0.417 | 8.95 | 12.3 | 12 | 1 | 1.07 | contended |
| ESCI (no repeated queries) | 1% | Ours | 10.1 | 0.897 | 0.948 | 8.46 | 8.33 | 0.554 | 9.14 | 12.6 | 12 | 1.02 | 1.11 | contended |
| ESCI (no repeated queries) | 3% | Ours | 14.6 | 0.904 | 0.946 | 8.46 | 8.47 | 0.879 | 9.58 | 13.1 | 12 | 0.999 | 1.17 | contended |
| ESCI (no repeated queries) | 10% | Ours | 23.8 | 0.901 | 0.942 | 8.46 | 7.86 | 1.11 | 9.8 | 13.2 | 12 | 1.08 | 1.31 | contended |
| ESCI (no repeated queries) | 0.3% | Ours (drift handling) (cutoff lost 4k requests) | 5.42 | 0.925 | 0.952 | 8.46 | 8.56 | 1.35 | 8.93 | 12.4 | 12 | 0.988 | 1.05 | contended |
| ESCI (no repeated queries) | 1% | Ours (drift handling) (cutoff lost 4k requests) (cutoff lost 1k requests) | 8.43 | 0.918 | 0.951 | 8.46 | 8.48 | 1.48 | 9.05 | 12.5 | 12 | 0.998 | 1.08 | contended |
| ESCI (no repeated queries) | 3% | Ours (drift handling) (cutoff lost 4k requests) (cutoff lost 1k requests) (cutoff lost 1k requests) | 12.9 | 0.915 | 0.949 | 8.46 | 8.57 | 1.31 | 9.46 | 12.9 | 12 | 0.987 | 1.13 | contended |
| ESCI (no repeated queries) | 10% | Ours (drift handling) (cutoff lost 4k requests) (cutoff lost 1k requests) (cutoff lost 1k requests) | 21.2 | 0.915 | 0.946 | 8.46 | 8.28 | 1.98 | 9.79 | 13.2 | 12 | 1.02 | 1.24 | contended |
| ESCI (no repeated queries) | 0.3% | Ours (continuous greedy) (cutoff lost 2k requests) | 6.23 | 0.925 | 0.952 | 8.46 | 8.59 | 1.3 | 9.02 | 12.5 | 12 | 0.985 | 1.06 | contended |
| ESCI (no repeated queries) | 1% | Ours (continuous greedy) (cutoff lost 2k requests) (cutoff lost 1k requests) | 8.86 | 0.927 | 0.951 | 8.46 | 8.59 | 1.67 | 9.2 | 12.6 | 12 | 0.985 | 1.09 | contended |
| ESCI (no repeated queries) | 3% | Ours (continuous greedy) (cutoff lost 2k requests) (cutoff lost 1k requests) | 15.2 | 0.912 | 0.948 | 8.46 | 8.39 | 1.83 | 9.41 | 12.8 | 12 | 1.01 | 1.16 | contended |
| ESCI (no repeated queries) | 10% | Ours (continuous greedy) (cutoff lost 2k requests) (cutoff lost 1k requests) | 24.5 | 0.915 | 0.945 | 8.46 | 8.05 | 1.99 | 9.81 | 13.3 | 12 | 1.05 | 1.28 | contended |
| ESCI (no repeated queries) | 0.3% | LRU (cutoff lost 103k requests) | 0.0633 | 0.65 | 0.955 | 8.46 | 9.75 | 0.443 | 9.76 | 25.9 | 12 | 0.868 | 1 | contended |
| ESCI (no repeated queries) | 1% | LRU (cutoff lost 103k requests) (cutoff lost 88k requests) | 0.816 | 0.929 | 0.955 | 8.46 | 10.3 | 1.96 | 10.4 | 32.1 | 12 | 0.822 | 1.01 | contended |
| ESCI (no repeated queries) | 3% | LRU (cutoff lost 103k requests) (cutoff lost 88k requests) (cutoff lost 3k requests) | 7.85 | 0.919 | 0.951 | 8.46 | 10.6 | 2.77 | 11.2 | 38.2 | 12 | 0.797 | 1.08 | contended |
| ESCI (no repeated queries) | 10% | LRU (cutoff lost 103k requests) (cutoff lost 88k requests) (cutoff lost 3k requests) | 17 | 0.914 | 0.948 | 8.46 | 10.7 | 3.52 | 12 | 35.7 | 12 | 0.791 | 1.18 | contended |
| ESCI (no repeated queries) | 0.3% | Aker | 0.00904 | 0.91 | 0.955 | 8.46 | 8.49 | 0.0318 | 8.5 | 12.1 | 12 | 0.996 | 1 | contended |
| ESCI (no repeated queries) | 1% | Aker | 0.0154 | 0.941 | 0.955 | 8.46 | 8.5 | 0.0426 | 8.51 | 12.1 | 12 | 0.995 | 1 | contended |
| ESCI (no repeated queries) | 3% | Aker | 0.0244 | 0.922 | 0.955 | 8.46 | 8.51 | 0.0637 | 8.52 | 12.1 | 12 | 0.994 | 1 | contended |
| ESCI (no repeated queries) | 10% | Aker | 0.112 | 0.934 | 0.955 | 8.46 | 8.52 | 0.0609 | 8.53 | 12.1 | 12 | 0.994 | 1 | contended |
| ESCI (no repeated queries) | 1% | Aker (paper default) | 0.103 | 0.74 | 0.955 | 8.46 | 8.49 | 0.0418 | 8.51 | 12.1 | 12 | 0.997 | 1 | contended |
| ESCI (no repeated queries) | 10% | Aker (paper default) | 0.958 | 0.711 | 0.952 | 8.46 | 8.44 | 0.0705 | 8.54 | 12.2 | 12 | 1 | 1.01 | contended |
| ESCI (no repeated queries) | 1% | QVCache (paper's D) | 1.76 | 0.293 | 0.943 | 8.46 | 8.87 | 0.475 | 9.02 | 12.5 | 12 | 0.954 | 1.02 | contended |
| ESCI (no repeated queries) | 10% | QVCache (paper's D) | 1.76 | 0.716 | 0.951 | 8.46 | 9.46 | 0.985 | 9.61 | 13.2 | 12 | 0.895 | 1.02 | contended |
| ESCI (no repeated queries) | 0.3% | QVCache | 0.175 | 0.473 | 0.954 | 8.46 | 8.95 | 0.469 | 8.96 | 12.5 | 12 | 0.946 | 1 | contended |
| ESCI (no repeated queries) | 1% | QVCache | 0.268 | 0.628 | 0.954 | 8.46 | 8.81 | 0.264 | 8.84 | 12.4 | 12 | 0.96 | 1 | contended |
| ESCI (no repeated queries) | 3% | QVCache | 0.406 | 0.773 | 0.954 | 8.46 | 8.94 | 0.402 | 8.96 | 12.5 | 12 | 0.947 | 1 | contended |
| ESCI (no repeated queries) | 10% | QVCache | 0.691 | 0.894 | 0.954 | 8.46 | 10.7 | 1.95 | 10.7 | 14.6 | 12 | 0.794 | 1.01 | contended |
| TriviaQA, Zipf 0.3 | 1% | Ours | 7 | 0.916 | 0.949 | 73.7 | 71 | 3.18 | 77.2 | 84.6 | 80.8 | 1.04 | 1.08 | contended |
| TriviaQA, Zipf 0.3 | 10% | Ours | 16.7 | 0.952 | 0.951 | 73.7 | 72.3 | 11.8 | 86 | 94.7 | 80.8 | 1.02 | 1.2 | contended |
| TriviaQA, Zipf 0.3 | 1% | Ours (drift handling) (cutoff lost 3k requests) | 6.46 | 0.918 | 0.949 | 73.7 | 72.5 | 9.65 | 77.5 | 84.9 | 80.8 | 1.02 | 1.06 | contended |
| TriviaQA, Zipf 0.3 | 10% | Ours (drift handling) (cutoff lost 3k requests) | 27.3 | 0.916 | 0.939 | 73.7 | 68.3 | 20.8 | 87.6 | 96.8 | 80.8 | 1.08 | 1.33 | contended |
| TriviaQA, Zipf 0.3 | 1% | Ours (continuous greedy) | 7.67 | 0.924 | 0.949 | 73.7 | 71.7 | 11.2 | 77 | 84.7 | 80.8 | 1.03 | 1.07 | contended |
| TriviaQA, Zipf 0.3 | 10% | Ours (continuous greedy) | 47.3 | 0.913 | 0.925 | 73.7 | 57.2 | 22.2 | 91 | 114 | 80.8 | 1.29 | 1.74 | contended |
| TriviaQA, Zipf 0.3 | 1% | LRU (cutoff lost 10k requests) | 3.16 | 0.922 | 0.952 | 73.7 | 75.4 | 8.61 | 77.9 | 86 | 80.8 | 0.977 | 1.03 | contended |
| TriviaQA, Zipf 0.3 | 10% | LRU (cutoff lost 10k requests) | 54.8 | 0.91 | 0.921 | 73.7 | 67.5 | 35.1 | 108 | 141 | 80.8 | 1.09 | 1.97 | contended |
| TriviaQA, Zipf 0.3 | 1% | Aker | 2.8 | 0.931 | 0.952 | 73.7 | 71.6 | 0.0411 | 73.7 | 80.9 | 80.8 | 1.03 | 1.03 | contended |
| TriviaQA, Zipf 0.3 | 10% | Aker | 15.3 | 0.958 | 0.948 | 73.7 | 62 | 0.034 | 73.8 | 80.9 | 80.8 | 1.19 | 1.18 | contended |
| TriviaQA, Zipf 0.3 | 1% | QVCache | 2.67 | 0.858 | 0.951 | 73.7 | 76.4 | 4.39 | 78.3 | 86.1 | 80.8 | 0.964 | 1.03 | contended |
| TriviaQA, Zipf 0.3 | 10% | QVCache | 15.6 | 0.909 | 0.943 | 73.7 | 63.7 | 0.834 | 74.6 | 81.9 | 80.8 | 1.16 | 1.18 | contended |
| TriviaQA, Zipf 0.6 | 1% | Ours | 16.5 | 0.915 | 0.943 | 74 | 65.1 | 3.1 | 77.4 | 84.8 | 81.6 | 1.14 | 1.2 | contended |
| TriviaQA, Zipf 0.6 | 10% | Ours | 7.67 | 0.968 | 0.953 | 74 | 80.7 | 11.6 | 86 | 95.9 | 81.6 | 0.917 | 1.08 | contended |
| TriviaQA, Zipf 0.6 | 1% | Ours (drift handling) | 16 | 0.916 | 0.943 | 74 | 67.1 | 12.3 | 77.6 | 85.3 | 81.6 | 1.1 | 1.17 | contended |
| TriviaQA, Zipf 0.6 | 10% | Ours (drift handling) (cutoff lost 1k requests) | 38.8 | 0.913 | 0.933 | 74 | 61.4 | 20.5 | 87.4 | 96.7 | 81.6 | 1.21 | 1.53 | contended |
| TriviaQA, Zipf 0.6 | 1% | Ours (continuous greedy) | 21.2 | 0.913 | 0.94 | 74 | 63.2 | 10.6 | 77.5 | 86.3 | 81.6 | 1.17 | 1.24 | contended |
| TriviaQA, Zipf 0.6 | 10% | Ours (continuous greedy) | 57.3 | 0.912 | 0.92 | 74 | 51.4 | 21.4 | 91.8 | 113 | 81.6 | 1.44 | 2.06 | contended |
| TriviaQA, Zipf 0.6 | 1% | LRU | 14.2 | 0.915 | 0.944 | 74 | 68 | 12.3 | 78.3 | 86.9 | 81.6 | 1.09 | 1.15 | contended |
| TriviaQA, Zipf 0.6 | 10% | LRU | 61.5 | 0.91 | 0.917 | 74 | 47.4 | 21.1 | 91.7 | 109 | 81.6 | 1.56 | 2.23 | contended |
| TriviaQA, Zipf 0.6 | 1% | Aker | 6.53 | 0.944 | 0.949 | 74 | 68.3 | 0.0284 | 73.3 | 81.6 | 81.6 | 1.08 | 1.07 | quiet |
| TriviaQA, Zipf 0.6 | 10% | Aker | 27.1 | 0.915 | 0.932 | 74 | 54.8 | 0.0357 | 74 | 80.7 | 81.6 | 1.35 | 1.37 | contended |
| TriviaQA, Zipf 0.6 | 1% | QVCache | 6.36 | 0.911 | 0.949 | 74 | 73.9 | 3.32 | 78.3 | 87 | 81.6 | 1 | 1.07 | quiet |
| TriviaQA, Zipf 0.6 | 10% | QVCache | 24.6 | 0.921 | 0.94 | 74 | 57.1 | 0.84 | 74.8 | 81.8 | 81.6 | 1.3 | 1.33 | contended |
| TriviaQA, Zipf 0.99 | 1% | Ours | 59.9 | 0.915 | 0.925 | 74.5 | 31.8 | 3.1 | 78.1 | 84.3 | 83.5 | 2.34 | 2.49 | contended |
| TriviaQA, Zipf 0.99 | 10% | Ours | 59.5 | 0.956 | 0.953 | 74.5 | 40.2 | 11.5 | 86.5 | 94.9 | 83.5 | 1.85 | 2.47 | contended |
| TriviaQA, Zipf 0.99 | 1% | Ours (drift handling) | 60.5 | 0.91 | 0.922 | 74.5 | 37.1 | 12.2 | 78.2 | 84.8 | 83.5 | 2.01 | 2.19 | contended |
| TriviaQA, Zipf 0.99 | 10% | Ours (drift handling) | 73.1 | 0.912 | 0.92 | 74.5 | 37.3 | 20.2 | 87.9 | 96.1 | 83.5 | 2 | 2.91 | contended |
| TriviaQA, Zipf 0.99 | 1% | Ours (continuous greedy) | 62.7 | 0.914 | 0.923 | 74.5 | 34.7 | 10.7 | 78.1 | 85.4 | 83.5 | 2.15 | 2.29 | contended |
| TriviaQA, Zipf 0.99 | 10% | Ours (continuous greedy) | 85.4 | 0.915 | 0.914 | 74.5 | 31.8 | 21.9 | 90.8 | 100 | 83.5 | 2.34 | 4.3 | contended |
| TriviaQA, Zipf 0.99 | 1% | LRU | 57.8 | 0.905 | 0.919 | 74.5 | 39.6 | 12.1 | 78.9 | 86.2 | 83.5 | 1.88 | 2.08 | contended |
| TriviaQA, Zipf 0.99 | 10% | LRU | 86.7 | 0.911 | 0.911 | 74.5 | 30.1 | 20.9 | 90 | 97.8 | 83.5 | 2.47 | 4.52 | contended |
| TriviaQA, Zipf 0.99 | 1% | Aker | 43.2 | 0.93 | 0.925 | 74.5 | 41.8 | 0.0346 | 74.6 | 81.9 | 83.5 | 1.78 | 1.76 | contended |
| TriviaQA, Zipf 0.99 | 10% | Aker | 59.2 | 0.937 | 0.918 | 74.5 | 29.2 | 0.0273 | 74.6 | 81.1 | 83.5 | 2.55 | 2.45 | contended |
| TriviaQA, Zipf 0.99 | 1% | QVCache | 59.4 | 0.903 | 0.918 | 74.5 | 33.7 | 4.97 | 79.7 | 86.8 | 83.5 | 2.21 | 2.46 | contended |
| TriviaQA, Zipf 0.99 | 10% | QVCache | 59.7 | 0.945 | 0.939 | 74.5 | 29.3 | 0.814 | 75.4 | 81.6 | 83.5 | 2.55 | 2.48 | contended |
| SPACEV real query log | 1% | Ours (drift handling) (cutoff lost 39k requests) | 2.55 | 0.934 | 0.913 | 64.3 | 64.8 | 12.5 | 66.1 | 78.5 | 77.6 | 0.993 | 1.02 | contended |
| SPACEV real query log | 10% | Ours (drift handling) (cutoff lost 39k requests) (cutoff lost 30k requests) | 4.29 | 0.915 | 0.914 | 64.3 | 71.9 | 17.7 | 74.2 | 86.5 | 77.6 | 0.895 | 1.04 | contended |
| SPACEV real query log | 1% | Ours (continuous greedy) (cutoff lost 20k requests) | 4.09 | 0.922 | 0.913 | 64.3 | 64.1 | 11.8 | 66.1 | 78.7 | 77.6 | 1 | 1.04 | contended |
| SPACEV real query log | 10% | Ours (continuous greedy) (cutoff lost 20k requests) (cutoff lost 45k requests) | 3.51 | 0.918 | 0.914 | 64.3 | 72.6 | 17.5 | 74.5 | 89.6 | 77.6 | 0.887 | 1.03 | contended |
| SPACEV real query log | 1% | LRU (cutoff lost 27k requests) | 2.56 | 0.932 | 0.913 | 64.3 | 67.2 | 11.6 | 68.8 | 110 | 77.6 | 0.957 | 1.02 | contended |
| SPACEV real query log | 10% | LRU (cutoff lost 27k requests) (cutoff lost 16k requests) | 6.35 | 0.921 | 0.914 | 64.3 | 73.3 | 24.7 | 76.8 | 105 | 77.6 | 0.877 | 1.06 | contended |
| SPACEV real query log | 1% | Aker | 3.25 | 0.944 | 0.911 | 64.3 | 61.9 | 0.0215 | 64.3 | 76.7 | 77.6 | 1.04 | 1.03 | contended |
| SPACEV real query log | 10% | Aker | 8.25 | 0.915 | 0.907 | 64.3 | 59.1 | 0.0233 | 64.3 | 76.3 | 77.6 | 1.09 | 1.09 | contended |
| SPACEV real query log | 1% | QVCache | 2.39 | 0.959 | 0.913 | 64.3 | 64.1 | 1.44 | 65.7 | 79 | 77.6 | 1 | 1.02 | contended |
| SPACEV real query log | 10% | QVCache | 7.54 | 0.738 | 0.897 | 64.3 | 60.1 | 0.461 | 64.7 | 77 | 77.6 | 1.07 | 1.08 | contended |

Sample-rate check (mean ms, cache + DiskANN / DiskANN alone / speedup; whole-run hit rate)

- Zipf 0.6, 1%, Ours: 1/20: pending; 1/10: pending; 1/5: pending
- Zipf 0.6, 1%, LRU: 1/20: pending; 1/10: pending; 1/5: pending
- Zipf 0.99, 10%, Ours: 1/20: pending; 1/10: pending; 1/5: pending
- Zipf 0.99, 10%, LRU: 1/20: pending; 1/10: pending; 1/5: pending
