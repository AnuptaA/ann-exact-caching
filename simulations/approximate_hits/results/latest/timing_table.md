| trace | budget | technique | hit % | ret. hits | recall all | DiskANN ms | cache+DiskANN ms | hit ms | miss ms | p99 ms | DiskANN p99 | speedup | 1/(1-h) | env |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPACEV, Zipf 0.3 | 0.3% | Ours | 0 | - | nan | 7.54 | 7.63 | - | 7.63 | 12 | 11.9 | 0.988 | 1 | composed |
| SPACEV, Zipf 0.3 | 1% | Ours | 0 | - | nan | 7.54 | 7.67 | - | 7.67 | 12.1 | 11.9 | 0.983 | 1 | composed |
| SPACEV, Zipf 0.3 | 3% | Ours | 0 | - | nan | 7.54 | 7.72 | - | 7.72 | 12.1 | 11.9 | 0.978 | 1 | composed |
| SPACEV, Zipf 0.3 | 10% | Ours | 11.3 | 0.918 | nan | 7.54 | 6.98 | 0.551 | 7.9 | 12.3 | 11.9 | 1.08 | 1.13 | composed |
| SPACEV, Zipf 0.3 | 0.3% | Ours (drift handling) | 1.1 | 0.93 | nan | 7.54 | 7.58 | 0.0978 | 7.64 | 12 | 11.9 | 0.995 | 1.01 | composed |
| SPACEV, Zipf 0.3 | 1% | Ours (drift handling) | 1.94 | 0.923 | nan | 7.54 | 7.56 | 0.542 | 7.68 | 12.1 | 11.9 | 0.998 | 1.02 | composed |
| SPACEV, Zipf 0.3 | 3% | Ours (drift handling) | 4.96 | 0.918 | nan | 7.54 | 7.35 | 0.349 | 7.75 | 12.1 | 11.9 | 1.03 | 1.05 | composed |
| SPACEV, Zipf 0.3 | 10% | Ours (drift handling) | 11.3 | 0.915 | nan | 7.54 | 6.98 | 0.542 | 7.89 | 12.2 | 11.9 | 1.08 | 1.13 | composed |
| SPACEV, Zipf 0.3 | 0.3% | Ours (continuous greedy) | 0.887 | 0.937 | nan | 7.54 | 7.56 | 0.101 | 7.65 | 12 | 11.9 | 0.997 | 1.01 | composed |
| SPACEV, Zipf 0.3 | 1% | Ours (continuous greedy) | 2.22 | 0.939 | nan | 7.54 | 7.52 | 0.796 | 7.74 | 12.2 | 11.9 | 1 | 1.02 | composed |
| SPACEV, Zipf 0.3 | 3% | Ours (continuous greedy) | 6.52 | 0.922 | nan | 7.54 | 7.26 | 0.46 | 7.8 | 12.3 | 11.9 | 1.04 | 1.07 | composed |
| SPACEV, Zipf 0.3 | 10% | Ours (continuous greedy) | 16.6 | 0.915 | nan | 7.54 | 6.61 | 0.724 | 8.01 | 12.3 | 11.9 | 1.14 | 1.2 | composed |
| SPACEV, Zipf 0.3 | 0.3% | LRU | 0.945 | 0.951 | nan | 7.54 | 7.7 | 0.28 | 7.76 | 12.1 | 11.9 | 0.98 | 1.01 | composed |
| SPACEV, Zipf 0.3 | 1% | LRU | 3.06 | 0.926 | nan | 7.54 | 7.82 | 1.44 | 8.03 | 12.5 | 11.9 | 0.965 | 1.03 | composed |
| SPACEV, Zipf 0.3 | 3% | LRU | 7.3 | 0.917 | nan | 7.54 | 7.96 | 1.24 | 8.45 | 12.8 | 11.9 | 0.947 | 1.08 | composed |
| SPACEV, Zipf 0.3 | 10% | LRU | 21.1 | 0.911 | nan | 7.54 | 7.28 | 1.57 | 8.78 | 13.1 | 11.9 | 1.04 | 1.27 | composed |
| SPACEV, Zipf 0.3 | 0.3% | Aker | 0.29 | 0.921 | nan | 7.54 | 7.57 | 0.0231 | 7.59 | 12 | 11.9 | 0.996 | 1 | composed |
| SPACEV, Zipf 0.3 | 1% | Aker | 0.886 | 0.926 | nan | 7.54 | 7.52 | 0.0276 | 7.58 | 12 | 11.9 | 1 | 1.01 | composed |
| SPACEV, Zipf 0.3 | 3% | Aker | 2.66 | 0.936 | nan | 7.54 | 7.43 | 0.0447 | 7.62 | 12 | 11.9 | 1.02 | 1.03 | composed |
| SPACEV, Zipf 0.3 | 10% | Aker | 7.05 | 0.899 | nan | 7.54 | 7.07 | 0.0278 | 7.6 | 12 | 11.9 | 1.07 | 1.08 | composed |
| SPACEV, Zipf 0.3 | 0.3% | Aker (paper default) | 1.29 | 0.731 | nan | 7.54 | 7.52 | 0.0984 | 7.62 | 11.9 | 11.9 | 1 | 1.01 | composed |
| SPACEV, Zipf 0.3 | 1% | Aker (paper default) | 3.29 | 0.761 | nan | 7.54 | 7.32 | 0.0306 | 7.58 | 11.9 | 11.9 | 1.03 | 1.03 | composed |
| SPACEV, Zipf 0.3 | 3% | Aker (paper default) | 7.61 | 0.798 | nan | 7.54 | 7.05 | 0.0794 | 7.63 | 11.9 | 11.9 | 1.07 | 1.08 | composed |
| SPACEV, Zipf 0.3 | 10% | Aker (paper default) | 9.03 | 0.812 | nan | 7.54 | 6.93 | 0.0277 | 7.58 | 11.9 | 11.9 | 1.09 | 1.1 | composed |
| SPACEV, Zipf 0.3 | 0.3% | QVCache | 0.211 | 0.941 | nan | 7.54 | 8 | 0.425 | 8.01 | 12.3 | 11.9 | 0.943 | 1 | composed |
| SPACEV, Zipf 0.3 | 1% | QVCache | 0.726 | 0.947 | nan | 7.54 | 8.21 | 0.656 | 8.24 | 12.6 | 11.9 | 0.919 | 1.01 | composed |
| SPACEV, Zipf 0.3 | 3% | QVCache | 3.94 | 0.92 | nan | 7.54 | 8.13 | 0.921 | 8.43 | 12.8 | 11.9 | 0.928 | 1.04 | composed |
| SPACEV, Zipf 0.3 | 10% | QVCache | 5.55 | 0.911 | nan | 7.54 | 7.29 | 0.137 | 7.69 | 12.1 | 11.9 | 1.03 | 1.06 | composed |
| SPACEV, Zipf 0.6 | 0.3% | Ours | 7.74 | 0.926 | nan | 7.11 | 6.74 | 0.179 | 7.26 | 11.3 | 11.1 | 1.06 | 1.08 | composed |
| SPACEV, Zipf 0.6 | 1% | Ours | 9.7 | 0.928 | nan | 7.11 | 6.72 | 0.617 | 7.31 | 11.3 | 11.1 | 1.06 | 1.11 | composed |
| SPACEV, Zipf 0.6 | 3% | Ours | 13.9 | 0.911 | nan | 7.11 | 6.53 | 0.962 | 7.37 | 11.4 | 11.1 | 1.09 | 1.16 | composed |
| SPACEV, Zipf 0.6 | 10% | Ours | 19.7 | 0.925 | nan | 7.11 | 6.24 | 0.846 | 7.55 | 11.5 | 11.1 | 1.14 | 1.24 | composed |
| SPACEV, Zipf 0.6 | 0.3% | Ours (drift handling) | 8.09 | 0.918 | nan | 7.11 | 6.73 | 0.176 | 7.25 | 11.3 | 11.1 | 1.06 | 1.09 | composed |
| SPACEV, Zipf 0.6 | 1% | Ours (drift handling) | 10 | 0.922 | nan | 7.11 | 6.71 | 0.616 | 7.31 | 11.3 | 11.1 | 1.06 | 1.11 | composed |
| SPACEV, Zipf 0.6 | 3% | Ours (drift handling) | 13 | 0.917 | nan | 7.11 | 6.58 | 0.967 | 7.37 | 11.4 | 11.1 | 1.08 | 1.15 | composed |
| SPACEV, Zipf 0.6 | 10% | Ours (drift handling) | 22.9 | 0.92 | nan | 7.11 | 6.13 | 1.01 | 7.61 | 11.6 | 11.1 | 1.16 | 1.3 | composed |
| SPACEV, Zipf 0.6 | 0.3% | Ours (continuous greedy) | 11.4 | 0.92 | nan | 7.11 | 6.56 | 0.304 | 7.3 | 11.3 | 11.1 | 1.08 | 1.13 | composed |
| SPACEV, Zipf 0.6 | 1% | Ours (continuous greedy) | 14.7 | 0.92 | nan | 7.11 | 6.5 | 0.998 | 7.38 | 11.3 | 11.1 | 1.09 | 1.17 | composed |
| SPACEV, Zipf 0.6 | 3% | Ours (continuous greedy) | 18.6 | 0.92 | nan | 7.11 | 6.31 | 1.16 | 7.42 | 11.3 | 11.1 | 1.13 | 1.23 | composed |
| SPACEV, Zipf 0.6 | 10% | Ours (continuous greedy) | 27.5 | 0.919 | nan | 7.11 | 5.85 | 1.08 | 7.67 | 11.4 | 11.1 | 1.22 | 1.38 | composed |
| SPACEV, Zipf 0.6 | 0.3% | LRU | 4.86 | 0.926 | nan | 7.11 | 7.05 | 0.437 | 7.37 | 11.4 | 11.1 | 1.01 | 1.05 | composed |
| SPACEV, Zipf 0.6 | 1% | LRU | 9.3 | 0.918 | nan | 7.11 | 7.11 | 1.43 | 7.62 | 11.6 | 11.1 | 1 | 1.1 | composed |
| SPACEV, Zipf 0.6 | 3% | LRU | 14.9 | 0.914 | nan | 7.11 | 7.04 | 1.86 | 7.9 | 11.9 | 11.1 | 1.01 | 1.18 | composed |
| SPACEV, Zipf 0.6 | 10% | LRU | 31.2 | 0.912 | nan | 7.11 | 6.27 | 1.81 | 8.25 | 12.1 | 11.1 | 1.14 | 1.45 | composed |
| SPACEV, Zipf 0.6 | 0.3% | Aker | 0.998 | 0.929 | nan | 7.11 | 7.12 | 0.0465 | 7.19 | 11.2 | 11.1 | 0.999 | 1.01 | composed |
| SPACEV, Zipf 0.6 | 1% | Aker | 3.76 | 0.89 | nan | 7.11 | 6.95 | 0.021 | 7.16 | 11.2 | 11.1 | 1.02 | 1.04 | composed |
| SPACEV, Zipf 0.6 | 3% | Aker | 8.13 | 0.909 | nan | 7.11 | 6.61 | 0.0464 | 7.21 | 11.2 | 11.1 | 1.08 | 1.09 | composed |
| SPACEV, Zipf 0.6 | 10% | Aker | 12.6 | 0.918 | nan | 7.11 | 6.31 | 0.0228 | 7.18 | 11.2 | 11.1 | 1.13 | 1.14 | composed |
| SPACEV, Zipf 0.6 | 0.3% | Aker (paper default) | 3.02 | 0.778 | nan | 7.11 | 6.98 | 0.0552 | 7.18 | 11.2 | 11.1 | 1.02 | 1.03 | composed |
| SPACEV, Zipf 0.6 | 1% | Aker (paper default) | 6.24 | 0.809 | nan | 7.11 | 6.69 | 0.0248 | 7.17 | 11.2 | 11.1 | 1.06 | 1.07 | composed |
| SPACEV, Zipf 0.6 | 3% | Aker (paper default) | 11.7 | 0.849 | nan | 7.11 | 6.33 | 0.0441 | 7.18 | 11.2 | 11.1 | 1.12 | 1.13 | composed |
| SPACEV, Zipf 0.6 | 10% | Aker (paper default) | 13.9 | 0.86 | nan | 7.11 | 6.15 | 0.022 | 7.17 | 11.1 | 11.1 | 1.16 | 1.16 | composed |
| SPACEV, Zipf 0.6 | 0.3% | QVCache | 1.45 | 0.957 | nan | 7.11 | 7.51 | 0.519 | 7.62 | 11.7 | 11.1 | 0.947 | 1.01 | composed |
| SPACEV, Zipf 0.6 | 1% | QVCache | 3.57 | 0.968 | nan | 7.11 | 7.61 | 0.739 | 7.85 | 11.9 | 11.1 | 0.934 | 1.04 | composed |
| SPACEV, Zipf 0.6 | 3% | QVCache | 10.4 | 0.922 | nan | 7.11 | 7.25 | 0.912 | 8 | 12 | 11.1 | 0.981 | 1.12 | composed |
| SPACEV, Zipf 0.6 | 10% | QVCache | 10.6 | 0.95 | nan | 7.11 | 6.77 | 0.435 | 7.53 | 11.6 | 11.1 | 1.05 | 1.12 | composed |
| SPACEV, Zipf 0.99 | 0.3% | Ours | 44.5 | 0.912 | nan | 7.24 | 4.24 | 0.165 | 7.59 | 11.8 | 12 | 1.7 | 1.8 | composed |
| SPACEV, Zipf 0.99 | 1% | Ours | 47.6 | 0.912 | nan | 7.24 | 4.06 | 0.291 | 7.58 | 11.8 | 12 | 1.78 | 1.91 | composed |
| SPACEV, Zipf 0.99 | 3% | Ours | 52.6 | 0.907 | nan | 7.24 | 3.84 | 0.625 | 7.65 | 11.8 | 12 | 1.88 | 2.11 | composed |
| SPACEV, Zipf 0.99 | 10% | Ours | 57 | 0.907 | nan | 7.24 | 3.82 | 0.894 | 7.77 | 11.8 | 12 | 1.89 | 2.32 | composed |
| SPACEV, Zipf 0.99 | 0.3% | Ours (drift handling) | 43.6 | 0.919 | nan | 7.24 | 4.32 | 0.165 | 7.58 | 11.8 | 12 | 1.68 | 1.77 | composed |
| SPACEV, Zipf 0.99 | 1% | Ours (drift handling) | 46.8 | 0.917 | nan | 7.24 | 4.13 | 0.296 | 7.58 | 11.8 | 12 | 1.75 | 1.88 | composed |
| SPACEV, Zipf 0.99 | 3% | Ours (drift handling) | 51.2 | 0.918 | nan | 7.24 | 3.94 | 0.627 | 7.65 | 11.8 | 12 | 1.84 | 2.05 | composed |
| SPACEV, Zipf 0.99 | 10% | Ours (drift handling) | 55.8 | 0.918 | nan | 7.24 | 3.89 | 0.89 | 7.77 | 11.8 | 12 | 1.86 | 2.26 | composed |
| SPACEV, Zipf 0.99 | 0.3% | Ours (continuous greedy) | 47.7 | 0.921 | nan | 7.24 | 4.08 | 0.265 | 7.66 | 11.8 | 12 | 1.78 | 1.91 | composed |
| SPACEV, Zipf 0.99 | 1% | Ours (continuous greedy) | 50.9 | 0.92 | nan | 7.24 | 3.9 | 0.391 | 7.57 | 11.8 | 12 | 1.85 | 2.04 | composed |
| SPACEV, Zipf 0.99 | 3% | Ours (continuous greedy) | 54.9 | 0.92 | nan | 7.24 | 3.85 | 0.77 | 7.7 | 11.8 | 12 | 1.88 | 2.22 | composed |
| SPACEV, Zipf 0.99 | 10% | Ours (continuous greedy) | 62.8 | 0.914 | nan | 7.24 | 3.65 | 1.07 | 7.91 | 11.8 | 12 | 1.98 | 2.69 | composed |
| SPACEV, Zipf 0.99 | 0.3% | LRU | 40.2 | 0.917 | nan | 7.24 | 4.66 | 0.337 | 7.71 | 12 | 12 | 1.55 | 1.67 | composed |
| SPACEV, Zipf 0.99 | 1% | LRU | 45.6 | 0.912 | nan | 7.24 | 4.43 | 0.526 | 7.74 | 12.1 | 12 | 1.63 | 1.84 | composed |
| SPACEV, Zipf 0.99 | 3% | LRU | 51.6 | 0.912 | nan | 7.24 | 4.31 | 0.952 | 7.89 | 12 | 12 | 1.68 | 2.07 | composed |
| SPACEV, Zipf 0.99 | 10% | LRU | 63.1 | 0.913 | nan | 7.24 | 4.01 | 1.47 | 8.24 | 11.9 | 12 | 1.8 | 2.71 | composed |
| SPACEV, Zipf 0.99 | 0.3% | Aker | 25 | 0.897 | nan | 7.24 | 5.49 | 0.0266 | 7.39 | 12 | 12 | 1.32 | 1.33 | composed |
| SPACEV, Zipf 0.99 | 1% | Aker | 34.2 | 0.914 | nan | 7.24 | 4.76 | 0.0143 | 7.38 | 11.8 | 12 | 1.52 | 1.52 | composed |
| SPACEV, Zipf 0.99 | 3% | Aker | 42.9 | 0.918 | nan | 7.24 | 4.11 | 0.0177 | 7.42 | 11.7 | 12 | 1.76 | 1.75 | composed |
| SPACEV, Zipf 0.99 | 10% | Aker | 44.9 | 0.912 | nan | 7.24 | 4.02 | 0.0177 | 7.43 | 11.7 | 12 | 1.8 | 1.81 | composed |
| SPACEV, Zipf 0.99 | 0.3% | Aker (paper default) | 22.3 | 0.923 | nan | 7.24 | 5.74 | 0.0251 | 7.36 | 12 | 12 | 1.26 | 1.29 | composed |
| SPACEV, Zipf 0.99 | 1% | Aker (paper default) | 31.7 | 0.939 | nan | 7.24 | 4.95 | 0.013 | 7.4 | 11.9 | 12 | 1.46 | 1.46 | composed |
| SPACEV, Zipf 0.99 | 3% | Aker (paper default) | 40.5 | 0.947 | nan | 7.24 | 4.3 | 0.0248 | 7.42 | 11.7 | 12 | 1.68 | 1.68 | composed |
| SPACEV, Zipf 0.99 | 10% | Aker (paper default) | 42.3 | 0.946 | nan | 7.24 | 4.15 | 0.011 | 7.39 | 11.7 | 12 | 1.74 | 1.73 | composed |
| SPACEV, Zipf 0.99 | 0.3% | QVCache | 33.4 | 0.917 | nan | 7.24 | 5.26 | 0.424 | 7.81 | 12.3 | 12 | 1.37 | 1.5 | composed |
| SPACEV, Zipf 0.99 | 1% | QVCache | 40.9 | 0.92 | nan | 7.24 | 4.91 | 0.669 | 8.03 | 12.3 | 12 | 1.47 | 1.69 | composed |
| SPACEV, Zipf 0.99 | 3% | QVCache | 41.2 | 0.974 | nan | 7.24 | 4.63 | 0.409 | 7.77 | 12.1 | 12 | 1.56 | 1.7 | composed |
| SPACEV, Zipf 0.99 | 10% | QVCache | 41.2 | 0.978 | nan | 7.24 | 4.53 | 0.307 | 7.67 | 12.1 | 12 | 1.6 | 1.7 | composed |
| ESCI (no repeated queries) | 0.3% | Ours | 6.71 | 0.894 | nan | 6.73 | 6.44 | 0.124 | 6.83 | 9.97 | 9.87 | 1.04 | 1.07 | composed |
| ESCI (no repeated queries) | 1% | Ours | 9.16 | 0.905 | nan | 6.73 | 6.34 | 0.295 | 6.89 | 10 | 9.87 | 1.06 | 1.1 | composed |
| ESCI (no repeated queries) | 3% | Ours | 12.6 | 0.899 | nan | 6.73 | 6.22 | 0.261 | 6.97 | 10.1 | 9.87 | 1.08 | 1.14 | composed |
| ESCI (no repeated queries) | 10% | Ours | 21.8 | 0.894 | nan | 6.73 | 5.76 | 0.479 | 7.09 | 10.1 | 9.87 | 1.17 | 1.28 | composed |
| ESCI (no repeated queries) | 0.3% | Ours (drift handling) | 5.41 | 0.926 | nan | 6.73 | 6.48 | 0.124 | 6.82 | 9.96 | 9.87 | 1.04 | 1.06 | composed |
| ESCI (no repeated queries) | 1% | Ours (drift handling) | 8.23 | 0.916 | nan | 6.73 | 6.41 | 0.292 | 6.88 | 10 | 9.87 | 1.05 | 1.09 | composed |
| ESCI (no repeated queries) | 3% | Ours (drift handling) | 11 | 0.915 | nan | 6.73 | 6.32 | 0.26 | 6.97 | 10.1 | 9.87 | 1.06 | 1.12 | composed |
| ESCI (no repeated queries) | 10% | Ours (drift handling) | 18.8 | 0.911 | nan | 6.73 | 5.93 | 0.461 | 7.07 | 10.1 | 9.87 | 1.13 | 1.23 | composed |
| ESCI (no repeated queries) | 0.3% | Ours (continuous greedy) | 6.16 | 0.922 | nan | 6.73 | 6.48 | 0.128 | 6.83 | 9.96 | 9.87 | 1.04 | 1.07 | composed |
| ESCI (no repeated queries) | 1% | Ours (continuous greedy) | 8.91 | 0.923 | nan | 6.73 | 6.42 | 0.376 | 6.91 | 10 | 9.87 | 1.05 | 1.1 | composed |
| ESCI (no repeated queries) | 3% | Ours (continuous greedy) | 13.4 | 0.917 | nan | 6.73 | 6.21 | 0.345 | 7 | 10.1 | 9.87 | 1.08 | 1.15 | composed |
| ESCI (no repeated queries) | 10% | Ours (continuous greedy) | 23.7 | 0.912 | nan | 6.73 | 5.75 | 0.685 | 7.17 | 10.2 | 9.87 | 1.17 | 1.31 | composed |
| ESCI (no repeated queries) | 0.3% | LRU | 0.0877 | 0.642 | nan | 6.73 | 6.84 | 0.131 | 6.85 | 10 | 9.87 | 0.984 | 1 | composed |
| ESCI (no repeated queries) | 1% | LRU | 4.75 | 0.922 | nan | 6.73 | 6.71 | 0.415 | 7 | 10.1 | 9.87 | 1 | 1.05 | composed |
| ESCI (no repeated queries) | 3% | LRU | 7.98 | 0.919 | nan | 6.73 | 6.83 | 0.571 | 7.33 | 10.5 | 9.87 | 0.986 | 1.09 | composed |
| ESCI (no repeated queries) | 10% | LRU | 15.6 | 0.913 | nan | 6.73 | 7.06 | 1.54 | 7.99 | 11.1 | 9.87 | 0.953 | 1.18 | composed |
| ESCI (no repeated queries) | 0.3% | Aker | 0.00904 | 0.91 | nan | 6.73 | 6.78 | 0.048 | 6.78 | 10.1 | 9.87 | 0.993 | 1 | composed |
| ESCI (no repeated queries) | 1% | Aker | 0.0154 | 0.941 | nan | 6.73 | 6.76 | 0.0288 | 6.77 | 10 | 9.87 | 0.995 | 1 | composed |
| ESCI (no repeated queries) | 3% | Aker | 0.0244 | 0.922 | nan | 6.73 | 6.82 | 0.101 | 6.82 | 10.1 | 9.87 | 0.987 | 1 | composed |
| ESCI (no repeated queries) | 10% | Aker | 0.112 | 0.934 | nan | 6.73 | 6.77 | 0.0443 | 6.78 | 9.98 | 9.87 | 0.995 | 1 | composed |
| ESCI (no repeated queries) | 0.3% | Aker (paper default) | 0.0325 | 0.733 | nan | 6.73 | 6.8 | 0.0561 | 6.8 | 10.1 | 9.87 | 0.99 | 1 | composed |
| ESCI (no repeated queries) | 1% | Aker (paper default) | 0.103 | 0.74 | nan | 6.73 | 6.75 | 0.0341 | 6.77 | 10 | 9.87 | 0.997 | 1 | composed |
| ESCI (no repeated queries) | 3% | Aker (paper default) | 0.312 | 0.707 | nan | 6.73 | 6.8 | 0.101 | 6.82 | 10.1 | 9.87 | 0.99 | 1 | composed |
| ESCI (no repeated queries) | 10% | Aker (paper default) | 0.958 | 0.711 | nan | 6.73 | 6.71 | 0.0523 | 6.79 | 9.99 | 9.87 | 1 | 1.01 | composed |
| ESCI (no repeated queries) | 0.3% | QVCache | 0.173 | 0.468 | nan | 6.73 | 7 | 0.244 | 7.01 | 10.2 | 9.87 | 0.961 | 1 | composed |
| ESCI (no repeated queries) | 1% | QVCache | 0.267 | 0.626 | nan | 6.73 | 6.87 | 0.116 | 6.89 | 10.1 | 9.87 | 0.98 | 1 | composed |
| ESCI (no repeated queries) | 3% | QVCache | 0.404 | 0.774 | nan | 6.73 | 6.92 | 0.178 | 6.94 | 10.1 | 9.87 | 0.973 | 1 | composed |
| ESCI (no repeated queries) | 10% | QVCache | 0.652 | 0.889 | nan | 6.73 | 7.79 | 0.994 | 7.83 | 11.1 | 9.87 | 0.864 | 1.01 | composed |
| TriviaQA, Zipf 0.3 | 1% | Ours | 5.78 | 0.921 | nan | 73.7 | 69.7 | 2.04 | 74.6 | 82.4 | 81.6 | 1.06 | 1.06 | composed |
| TriviaQA, Zipf 0.3 | 10% | Ours | 29.9 | 0.904 | nan | 73.7 | 56.3 | 12.1 | 76.8 | 84.3 | 81.6 | 1.31 | 1.43 | composed |
| TriviaQA, Zipf 0.3 | 1% | Ours (drift handling) | 6.37 | 0.916 | nan | 73.7 | 69.6 | 2.04 | 74.6 | 82.5 | 81.6 | 1.06 | 1.07 | composed |
| TriviaQA, Zipf 0.3 | 10% | Ours (drift handling) | 28.9 | 0.906 | nan | 73.7 | 56.8 | 12 | 76.8 | 84.2 | 81.6 | 1.3 | 1.41 | composed |
| TriviaQA, Zipf 0.3 | 1% | Ours (continuous greedy) | 9.28 | 0.919 | nan | 73.7 | 67.8 | 3.28 | 75 | 82.8 | 81.6 | 1.09 | 1.1 | composed |
| TriviaQA, Zipf 0.3 | 10% | Ours (continuous greedy) | 49 | 0.908 | nan | 73.7 | 46.1 | 14.4 | 77.9 | 84.9 | 81.6 | 1.6 | 1.96 | composed |
| TriviaQA, Zipf 0.3 | 1% | LRU | 3.25 | 0.906 | nan | 73.7 | 73 | 4.64 | 75.8 | 83.7 | 81.6 | 1.01 | 1.03 | composed |
| TriviaQA, Zipf 0.3 | 10% | LRU | 53.9 | 0.906 | nan | 73.7 | 44.7 | 15.8 | 79.8 | 86.7 | 81.6 | 1.65 | 2.17 | composed |
| TriviaQA, Zipf 0.3 | 1% | Aker | 2.8 | 0.931 | nan | 73.7 | 71.7 | 0.0313 | 73.8 | 81.7 | 81.6 | 1.03 | 1.03 | composed |
| TriviaQA, Zipf 0.3 | 10% | Aker | 15.3 | 0.958 | nan | 73.7 | 62 | 0.0248 | 73.8 | 81.6 | 81.6 | 1.19 | 1.18 | composed |
| TriviaQA, Zipf 0.3 | 1% | QVCache | 2.21 | 0.831 | nan | 73.7 | 74.7 | 2.19 | 75.9 | 84 | 81.6 | 0.987 | 1.02 | composed |
| TriviaQA, Zipf 0.3 | 10% | QVCache | 15.6 | 0.87 | nan | 73.7 | 63.6 | 0.859 | 74.6 | 82.4 | 81.6 | 1.16 | 1.18 | composed |
| TriviaQA, Zipf 0.6 | 1% | Ours | 15 | 0.918 | nan | 73.5 | 63.8 | 2.07 | 74.5 | 82.7 | 81.9 | 1.15 | 1.18 | composed |
| TriviaQA, Zipf 0.6 | 10% | Ours | 40.7 | 0.909 | nan | 73.5 | 49.6 | 11.7 | 75.7 | 83.9 | 81.9 | 1.48 | 1.68 | composed |
| TriviaQA, Zipf 0.6 | 1% | Ours (drift handling) | 15.3 | 0.916 | nan | 73.5 | 63.6 | 2.08 | 74.5 | 82.7 | 81.9 | 1.15 | 1.18 | composed |
| TriviaQA, Zipf 0.6 | 10% | Ours (drift handling) | 41 | 0.907 | nan | 73.5 | 49.5 | 11.7 | 75.7 | 83.5 | 81.9 | 1.48 | 1.7 | composed |
| TriviaQA, Zipf 0.6 | 1% | Ours (continuous greedy) | 20.8 | 0.919 | nan | 73.5 | 59.5 | 3.5 | 74.9 | 82.8 | 81.9 | 1.23 | 1.26 | composed |
| TriviaQA, Zipf 0.6 | 10% | Ours (continuous greedy) | 59.5 | 0.907 | nan | 73.5 | 38.6 | 13.3 | 76.4 | 83.7 | 81.9 | 1.9 | 2.47 | composed |
| TriviaQA, Zipf 0.6 | 1% | LRU | 14.6 | 0.912 | nan | 73.5 | 64.2 | 4.77 | 75.9 | 84.1 | 81.9 | 1.14 | 1.17 | composed |
| TriviaQA, Zipf 0.6 | 10% | LRU | 61 | 0.908 | nan | 73.5 | 38.7 | 14.5 | 78.5 | 85.6 | 81.9 | 1.9 | 2.57 | composed |
| TriviaQA, Zipf 0.6 | 1% | Aker | 6.53 | 0.944 | nan | 73.5 | 68.5 | 0.0239 | 73.5 | 81.8 | 81.9 | 1.07 | 1.07 | composed |
| TriviaQA, Zipf 0.6 | 10% | Aker | 27.1 | 0.915 | nan | 73.5 | 54.4 | 0.0277 | 73.5 | 81.3 | 81.9 | 1.35 | 1.37 | composed |
| TriviaQA, Zipf 0.6 | 1% | QVCache | 6.5 | 0.922 | nan | 73.5 | 71.5 | 2.03 | 75.6 | 84.3 | 81.9 | 1.03 | 1.07 | composed |
| TriviaQA, Zipf 0.6 | 10% | QVCache | 24.6 | 0.885 | nan | 73.5 | 56.7 | 0.824 | 74.2 | 82.2 | 81.9 | 1.3 | 1.33 | composed |
| TriviaQA, Zipf 0.99 | 1% | Ours | 59.9 | 0.911 | nan | 74.3 | 30.5 | 2.4 | 75.6 | 82.4 | 84.6 | 2.44 | 2.5 | composed |
| TriviaQA, Zipf 0.99 | 10% | Ours | 74.8 | 0.907 | nan | 74.3 | 26.8 | 11.4 | 76.4 | 82.6 | 84.6 | 2.77 | 3.96 | composed |
| TriviaQA, Zipf 0.99 | 1% | Ours (drift handling) | 60.1 | 0.91 | nan | 74.3 | 30.7 | 2.41 | 75.6 | 82.4 | 84.6 | 2.42 | 2.5 | composed |
| TriviaQA, Zipf 0.99 | 10% | Ours (drift handling) | 75.1 | 0.906 | nan | 74.3 | 26.6 | 11.2 | 76.4 | 82.7 | 84.6 | 2.8 | 4.02 | composed |
| TriviaQA, Zipf 0.99 | 1% | Ours (continuous greedy) | 63.4 | 0.911 | nan | 74.3 | 28.7 | 3.61 | 76 | 82.4 | 84.6 | 2.59 | 2.73 | composed |
| TriviaQA, Zipf 0.99 | 10% | Ours (continuous greedy) | 85.8 | 0.908 | nan | 74.3 | 21.1 | 12.2 | 76.8 | 82 | 84.6 | 3.52 | 7.03 | composed |
| TriviaQA, Zipf 0.99 | 1% | LRU | 58.3 | 0.906 | nan | 74.3 | 33.2 | 4.14 | 76.5 | 82.9 | 84.6 | 2.24 | 2.4 | composed |
| TriviaQA, Zipf 0.99 | 10% | LRU | 86.7 | 0.907 | nan | 74.3 | 21.4 | 13.3 | 78.4 | 82.8 | 84.6 | 3.47 | 7.55 | composed |
| TriviaQA, Zipf 0.99 | 1% | Aker | 43.2 | 0.93 | nan | 74.3 | 41.8 | 0.0241 | 74.5 | 82.8 | 84.6 | 1.78 | 1.76 | composed |
| TriviaQA, Zipf 0.99 | 10% | Aker | 59.2 | 0.937 | nan | 74.3 | 29.1 | 0.0205 | 74.5 | 81.6 | 84.6 | 2.55 | 2.45 | composed |
| TriviaQA, Zipf 0.99 | 1% | QVCache | 34.7 | 0.954 | nan | 74.3 | 49.5 | 2 | 76.5 | 85.5 | 84.6 | 1.5 | 1.53 | composed |
| TriviaQA, Zipf 0.99 | 10% | QVCache | 59.7 | 0.916 | nan | 74.3 | 28.8 | 0.384 | 74.8 | 81.7 | 84.6 | 2.59 | 2.48 | composed |
| SPACEV real query log | 1% | Ours (drift handling) | 3.85 | 0.931 | nan | 130 | 125 | 1.29 | 130 | 143 | 142 | 1.03 | 1.04 | composed |
| SPACEV real query log | 10% | Ours (drift handling) | 6.06 | 0.917 | nan | 130 | 125 | 5.85 | 132 | 145 | 142 | 1.04 | 1.06 | composed |
| SPACEV real query log | 1% | Ours (continuous greedy) | 5.18 | 0.927 | nan | 130 | 124 | 1.42 | 130 | 143 | 142 | 1.05 | 1.05 | composed |
| SPACEV real query log | 10% | Ours (continuous greedy) | 8.41 | 0.911 | nan | 130 | 123 | 6.56 | 132 | 145 | 142 | 1.06 | 1.09 | composed |
| SPACEV real query log | 1% | LRU | 3.3 | 0.934 | nan | 130 | 126 | 1.8 | 130 | 143 | 142 | 1.03 | 1.03 | composed |
| SPACEV real query log | 10% | LRU | 8.59 | 0.917 | nan | 130 | 123 | 7.71 | 133 | 146 | 142 | 1.06 | 1.09 | composed |
| SPACEV real query log | 1% | Aker | 3.25 | 0.945 | nan | 130 | 125 | 0.0172 | 130 | 142 | 142 | 1.04 | 1.03 | composed |
| SPACEV real query log | 10% | Aker | 8.25 | 0.916 | nan | 130 | 119 | 0.0194 | 130 | 142 | 142 | 1.09 | 1.09 | composed |
| SPACEV real query log | 1% | QVCache | 2.73 | 0.978 | nan | 130 | 127 | 0.983 | 130 | 143 | 142 | 1.02 | 1.03 | composed |
| SPACEV real query log | 10% | QVCache | 7.54 | 0.751 | nan | 130 | 121 | 0.83 | 130 | 143 | 142 | 1.07 | 1.08 | composed |
| TripClick real health-search log | 0.3% | Ours (drift handling) | 1.97 | 0.922 | nan | 9.42 | 9.45 | 0.24 | 9.62 | 13.8 | 13.6 | 0.997 | 1.02 | composed |
| TripClick real health-search log | 1% | Ours (drift handling) | 4.18 | 0.925 | nan | 9.42 | 9.39 | 0.477 | 9.79 | 14 | 13.6 | 1 | 1.04 | composed |
| TripClick real health-search log | 3% | Ours (drift handling) | 7.55 | 0.918 | nan | 9.42 | 9.49 | 0.936 | 10.1 | 14 | 13.6 | 0.993 | 1.08 | composed |
| TripClick real health-search log | 10% | Ours (drift handling) | 13.4 | 0.915 | nan | 9.42 | 9.53 | 1.59 | 10.7 | 14.7 | 13.6 | 0.988 | 1.16 | composed |
| TripClick real health-search log | 0.3% | Ours (continuous greedy) | 4.33 | 0.928 | nan | 9.42 | 9.16 | 0.212 | 9.6 | 13.9 | 13.6 | 1.03 | 1.05 | composed |
| TripClick real health-search log | 1% | Ours (continuous greedy) | 6.45 | 0.925 | nan | 9.42 | 9.13 | 0.396 | 9.73 | 14 | 13.6 | 1.03 | 1.07 | composed |
| TripClick real health-search log | 3% | Ours (continuous greedy) | 10.4 | 0.921 | nan | 9.42 | 8.94 | 0.748 | 9.87 | 14.2 | 13.6 | 1.05 | 1.12 | composed |
| TripClick real health-search log | 10% | Ours (continuous greedy) | 25.1 | 0.912 | nan | 9.42 | 8.42 | 1.54 | 10.7 | 14.9 | 13.6 | 1.12 | 1.34 | composed |
| TripClick real health-search log | 0.3% | LRU | 14.6 | 0.912 | nan | 9.42 | 8.36 | 0.251 | 9.68 | 13.7 | 13.6 | 1.13 | 1.17 | composed |
| TripClick real health-search log | 1% | LRU | 17.4 | 0.914 | nan | 9.42 | 8.42 | 0.659 | 9.99 | 14 | 13.6 | 1.12 | 1.21 | composed |
| TripClick real health-search log | 3% | LRU | 22.8 | 0.912 | nan | 9.42 | 8.31 | 1.24 | 10.4 | 14.6 | 13.6 | 1.13 | 1.3 | composed |
| TripClick real health-search log | 10% | LRU | 32.6 | 0.91 | nan | 9.42 | 9.31 | 3.08 | 12.2 | 16.4 | 13.6 | 1.01 | 1.48 | composed |
| TripClick real health-search log | 0.3% | Aker | 9.22 | 0.914 | nan | 9.42 | 8.69 | 0.0912 | 9.54 | 13.5 | 13.6 | 1.08 | 1.1 | composed |
| TripClick real health-search log | 1% | Aker | 11.1 | 0.925 | nan | 9.42 | 8.56 | 0.0448 | 9.51 | 13.5 | 13.6 | 1.1 | 1.12 | composed |
| TripClick real health-search log | 3% | Aker | 13.7 | 0.927 | nan | 9.42 | 8.36 | 0.0611 | 9.54 | 13.6 | 13.6 | 1.13 | 1.16 | composed |
| TripClick real health-search log | 10% | Aker | 18.3 | 0.937 | nan | 9.42 | 7.84 | 0.0511 | 9.56 | 13.5 | 13.6 | 1.2 | 1.22 | composed |
| TripClick real health-search log | 0.3% | QVCache | 6.57 | 0.798 | nan | 9.42 | 9.21 | 0.334 | 9.79 | 13.8 | 13.6 | 1.02 | 1.07 | composed |
| TripClick real health-search log | 1% | QVCache | 8.33 | 0.783 | nan | 9.42 | 9.87 | 1.06 | 10.6 | 14.6 | 13.6 | 0.954 | 1.09 | composed |
| TripClick real health-search log | 3% | QVCache | 10.4 | 0.801 | nan | 9.42 | 9.37 | 0.764 | 10.3 | 14.3 | 13.6 | 1.01 | 1.12 | composed |
| TripClick real health-search log | 10% | QVCache | 13.9 | 0.874 | nan | 9.42 | 10.3 | 1.95 | 11.5 | 15.6 | 13.6 | 0.917 | 1.16 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | Ours | 42.5 | 0.922 | nan | 11.5 | 6.77 | 0.145 | 11.8 | 14.9 | 15.1 | 1.7 | 1.74 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | Ours | 56.7 | 0.905 | nan | 11.5 | 4.89 | 0.298 | 12 | 14.8 | 15.1 | 2.35 | 2.31 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | Ours | 63.3 | 0.927 | nan | 11.5 | 4.81 | 0.722 | 12.2 | 15 | 15.1 | 2.39 | 2.72 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | Ours | 64.2 | 0.931 | nan | 11.5 | 5 | 0.93 | 12.3 | 15.1 | 15.1 | 2.3 | 2.79 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | Ours (drift handling) | 42.6 | 0.919 | nan | 11.5 | 6.61 | 0.146 | 11.8 | 14.8 | 15.1 | 1.74 | 1.74 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | Ours (drift handling) | 53.7 | 0.916 | nan | 11.5 | 5.39 | 0.275 | 12 | 14.8 | 15.1 | 2.13 | 2.16 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | Ours (drift handling) | 63.4 | 0.926 | nan | 11.5 | 4.86 | 0.771 | 12.3 | 15 | 15.1 | 2.36 | 2.73 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | Ours (drift handling) | 63.9 | 0.935 | nan | 11.5 | 5.07 | 0.935 | 12.3 | 15.1 | 15.1 | 2.27 | 2.77 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | Ours (continuous greedy) | 49.5 | 0.92 | nan | 11.5 | 6.14 | 0.215 | 11.8 | 14.9 | 15.1 | 1.87 | 1.98 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | Ours (continuous greedy) | 65.6 | 0.913 | nan | 11.5 | 4.03 | 0.427 | 11.7 | 14.3 | 15.1 | 2.85 | 2.91 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | Ours (continuous greedy) | 79.5 | 0.905 | nan | 11.5 | 3.29 | 1.12 | 12.3 | 14.5 | 15.1 | 3.49 | 4.87 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | Ours (continuous greedy) | 88.4 | 0.915 | nan | 11.5 | 2.64 | 1.38 | 12.6 | 14.2 | 15.1 | 4.35 | 8.6 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | LRU | 35.1 | 0.919 | nan | 11.5 | 7.61 | 0.241 | 11.8 | 14.9 | 15.1 | 1.51 | 1.54 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | LRU | 57.4 | 0.913 | nan | 11.5 | 5.35 | 0.544 | 12 | 15.1 | 15.1 | 2.15 | 2.35 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | LRU | 77.3 | 0.897 | nan | 11.5 | 3.95 | 1.39 | 12.8 | 15.7 | 15.1 | 2.91 | 4.4 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | LRU | 93 | 0.901 | nan | 11.5 | 2.38 | 1.48 | 12.6 | 13.8 | 15.1 | 4.83 | 14.2 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | Aker | 1.76 | 0.899 | nan | 11.5 | 11.5 | 0.0394 | 11.5 | 15.2 | 15.1 | 1 | 1.02 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | Aker | 4.9 | 0.904 | nan | 11.5 | 11.3 | 0.0854 | 11.6 | 15.2 | 15.1 | 1.02 | 1.05 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | Aker | 16.7 | 0.904 | nan | 11.5 | 9.79 | 0.0835 | 11.6 | 15.2 | 15.1 | 1.17 | 1.2 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | Aker | 16.7 | 0.904 | nan | 11.5 | 10.1 | 0.0762 | 11.9 | 16.2 | 15.1 | 1.14 | 1.2 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | QVCache | 18.3 | 0.762 | nan | 11.5 | 9.93 | 0.185 | 11.7 | 15.3 | 15.1 | 1.16 | 1.22 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | QVCache | 29.7 | 0.9 | nan | 11.5 | 8.72 | 0.766 | 12.2 | 15.4 | 15.1 | 1.32 | 1.42 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | QVCache | 46.4 | 0.956 | nan | 11.5 | 6.61 | 0.264 | 11.7 | 14.9 | 15.1 | 1.74 | 1.86 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | QVCache | 46.4 | 0.956 | nan | 11.5 | 6.53 | 0.183 | 11.6 | 14.8 | 15.1 | 1.76 | 1.86 | composed |
| ESCI, DiskANN recall 0.97 | 1% | Ours | 9.1 | 0.903 | nan | 11.7 | 10.9 | 0.179 | 11.8 | 15.5 | 15.4 | 1.07 | 1.1 | composed |
| ESCI, DiskANN recall 0.97 | 10% | Ours | 22.2 | 0.898 | nan | 11.7 | 9.66 | 0.473 | 12 | 15.6 | 15.4 | 1.21 | 1.29 | composed |
| ESCI, DiskANN recall 0.97 | 1% | Ours (drift handling) | 8.03 | 0.918 | nan | 11.7 | 11 | 0.182 | 11.8 | 15.4 | 15.4 | 1.06 | 1.09 | composed |
| ESCI, DiskANN recall 0.97 | 10% | Ours (drift handling) | 20.3 | 0.909 | nan | 11.7 | 9.83 | 0.47 | 12 | 15.6 | 15.4 | 1.19 | 1.26 | composed |
| ESCI, DiskANN recall 0.97 | 1% | Ours (continuous greedy) | 8.99 | 0.922 | nan | 11.7 | 11 | 0.223 | 11.9 | 15.5 | 15.4 | 1.06 | 1.1 | composed |
| ESCI, DiskANN recall 0.97 | 10% | Ours (continuous greedy) | 25 | 0.911 | nan | 11.7 | 9.45 | 0.648 | 12.1 | 15.6 | 15.4 | 1.24 | 1.33 | composed |
| ESCI, DiskANN recall 0.97 | 1% | LRU | 4.73 | 0.924 | nan | 11.7 | 11.5 | 0.259 | 12 | 15.6 | 15.4 | 1.02 | 1.05 | composed |
| ESCI, DiskANN recall 0.97 | 10% | LRU | 16.3 | 0.913 | nan | 11.7 | 11 | 1.37 | 12.8 | 16.1 | 15.4 | 1.06 | 1.2 | composed |
| ESCI, DiskANN recall 0.97 | 1% | Aker | 0.0145 | 0.919 | nan | 11.7 | 11.8 | 0.0418 | 11.8 | 15.5 | 15.4 | 0.995 | 1 | composed |
| ESCI, DiskANN recall 0.97 | 10% | Aker | 0.102 | 0.925 | nan | 11.7 | 11.8 | 0.0949 | 11.8 | 15.5 | 15.4 | 0.995 | 1 | composed |
| ESCI, DiskANN recall 0.97 | 1% | QVCache | 0.267 | 0.624 | nan | 11.7 | 11.8 | 0.117 | 11.9 | 15.6 | 15.4 | 0.988 | 1 | composed |
| ESCI, DiskANN recall 0.97 | 10% | QVCache | 0.653 | 0.904 | nan | 11.7 | 12.8 | 1.01 | 12.8 | 16.5 | 15.4 | 0.917 | 1.01 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 1% | Ours | 10.9 | 0.913 | nan | 18.2 | 16.6 | 0.405 | 18.3 | 23.5 | 23.4 | 1.09 | 1.12 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 10% | Ours | 19.7 | 0.926 | nan | 18.2 | 15.1 | 0.867 | 18.6 | 23.7 | 23.4 | 1.2 | 1.25 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 1% | Ours (drift handling) | 10.3 | 0.919 | nan | 18.2 | 16.7 | 0.389 | 18.3 | 23.5 | 23.4 | 1.09 | 1.11 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 10% | Ours (drift handling) | 20.3 | 0.92 | nan | 18.2 | 14.9 | 0.861 | 18.6 | 23.6 | 23.4 | 1.22 | 1.25 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 1% | Ours (continuous greedy) | 14.7 | 0.921 | nan | 18.2 | 15.9 | 0.623 | 18.4 | 23.4 | 23.4 | 1.14 | 1.17 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 10% | Ours (continuous greedy) | 28.5 | 0.915 | nan | 18.2 | 13.6 | 1.07 | 18.7 | 23.8 | 23.4 | 1.33 | 1.4 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 1% | LRU | 9.57 | 0.92 | nan | 18.2 | 17.1 | 0.99 | 18.7 | 23.8 | 23.4 | 1.06 | 1.11 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 10% | LRU | 31.6 | 0.915 | nan | 18.2 | 13.8 | 1.8 | 19.3 | 24.1 | 23.4 | 1.32 | 1.46 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 1% | Aker | 3.77 | 0.892 | nan | 18.2 | 17.6 | 0.0461 | 18.2 | 23.4 | 23.4 | 1.03 | 1.04 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 10% | Aker | 12.6 | 0.92 | nan | 18.2 | 16.1 | 0.061 | 18.3 | 23.5 | 23.4 | 1.13 | 1.14 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 1% | QVCache | 3.34 | 0.968 | nan | 18.2 | 18.2 | 0.757 | 18.9 | 24.1 | 23.4 | 0.996 | 1.03 | composed |
| SPACEV Zipf 0.6, DiskANN recall 0.97 | 10% | QVCache | 10.6 | 0.952 | nan | 18.2 | 16.6 | 0.441 | 18.6 | 23.6 | 23.4 | 1.09 | 1.12 | composed |

Sample-rate check (mean ms, cache + DiskANN / DiskANN alone / speedup; whole-run hit rate)

- Zipf 0.6, 1%, Ours: 1/20: 6.72 / 7.11 / 1.059 (h 9.7%, n 1600); 1/10: 6.72 / 7.11 / 1.059 (h 9.7%, n 1600); 1/5: 6.72 / 7.11 / 1.059 (h 9.7%, n 1600)
- Zipf 0.6, 1%, LRU: 1/20: 7.11 / 7.11 / 1.001 (h 9.3%, n 1600); 1/10: 7.11 / 7.11 / 1.001 (h 9.3%, n 1600); 1/5: 7.11 / 7.11 / 1.001 (h 9.3%, n 1600)
- Zipf 0.99, 10%, Ours: 1/20: 3.82 / 7.24 / 1.892 (h 57.0%, n 1598); 1/10: 3.82 / 7.24 / 1.892 (h 57.0%, n 1598); 1/5: 3.82 / 7.24 / 1.892 (h 57.0%, n 1598)
- Zipf 0.99, 10%, LRU: 1/20: 4.01 / 7.24 / 1.804 (h 63.1%, n 1598); 1/10: 4.01 / 7.24 / 1.804 (h 63.1%, n 1598); 1/5: 4.01 / 7.24 / 1.804 (h 63.1%, n 1598)
