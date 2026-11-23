| trace | budget | technique | hit % | ret. hits | recall all | DiskANN ms | cache+DiskANN ms | hit ms | miss ms | p99 ms | DiskANN p99 | speedup | 1/(1-h) | env |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SPACEV, Zipf 0.3 | 0.3% | Ours, final config | 2.49 | 0.93 | nan | 7.54 | 7.56 | 0.247 | 7.78 | 12.2 | 11.9 | 0.998 | 1.03 | composed |
| SPACEV, Zipf 0.3 | 1% | Ours, final config | 7.3 | 0.923 | nan | 7.54 | 7.47 | 1.48 | 7.97 | 12.6 | 11.9 | 1.01 | 1.08 | composed |
| SPACEV, Zipf 0.3 | 3% | Ours, final config | 18.3 | 0.916 | nan | 7.54 | 6.65 | 0.726 | 8.01 | 12.3 | 11.9 | 1.13 | 1.22 | composed |
| SPACEV, Zipf 0.3 | 10% | Ours, final config | 23.3 | 0.914 | nan | 7.54 | 6.31 | 0.737 | 8.06 | 12.4 | 11.9 | 1.2 | 1.3 | composed |
| SPACEV, Zipf 0.3 | 0.3% | Ours | 0 | - | nan | 7.54 | 7.63 | - | 7.63 | 12 | 11.9 | 0.988 | 1 | composed |
| SPACEV, Zipf 0.3 | 1% | Ours | 0 | - | nan | 7.54 | 7.67 | - | 7.67 | 12.1 | 11.9 | 0.983 | 1 | composed |
| SPACEV, Zipf 0.3 | 3% | Ours | 0 | - | nan | 7.54 | 7.72 | - | 7.72 | 12.1 | 11.9 | 0.978 | 1 | composed |
| SPACEV, Zipf 0.3 | 10% | Ours | 11.3 | 0.918 | nan | 7.54 | 6.98 | 0.551 | 7.9 | 12.3 | 11.9 | 1.08 | 1.13 | composed |
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
| SPACEV, Zipf 0.6 | 0.3% | Ours, final config | 11.5 | 0.923 | nan | 7.11 | 6.61 | 0.379 | 7.34 | 11.3 | 11.1 | 1.08 | 1.13 | composed |
| SPACEV, Zipf 0.6 | 1% | Ours, final config | 18.1 | 0.915 | nan | 7.11 | 6.37 | 1.17 | 7.46 | 11.4 | 11.1 | 1.12 | 1.22 | composed |
| SPACEV, Zipf 0.6 | 3% | Ours, final config | 29.5 | 0.909 | nan | 7.11 | 5.94 | 1.57 | 7.62 | 11.5 | 11.1 | 1.2 | 1.42 | composed |
| SPACEV, Zipf 0.6 | 10% | Ours, final config | 32.3 | 0.908 | nan | 7.11 | 5.51 | 0.832 | 7.58 | 11.4 | 11.1 | 1.29 | 1.48 | composed |
| SPACEV, Zipf 0.6 | 0.3% | Ours | 7.74 | 0.926 | nan | 7.11 | 6.74 | 0.179 | 7.26 | 11.3 | 11.1 | 1.06 | 1.08 | composed |
| SPACEV, Zipf 0.6 | 1% | Ours | 9.7 | 0.928 | nan | 7.11 | 6.72 | 0.617 | 7.31 | 11.3 | 11.1 | 1.06 | 1.11 | composed |
| SPACEV, Zipf 0.6 | 3% | Ours | 13.9 | 0.911 | nan | 7.11 | 6.53 | 0.962 | 7.37 | 11.4 | 11.1 | 1.09 | 1.16 | composed |
| SPACEV, Zipf 0.6 | 10% | Ours | 19.7 | 0.925 | nan | 7.11 | 6.24 | 0.846 | 7.55 | 11.5 | 11.1 | 1.14 | 1.24 | composed |
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
| SPACEV, Zipf 0.99 | 0.3% | Ours, final config | 46.6 | 0.926 | nan | 7.24 | 4.16 | 0.229 | 7.74 | 11.9 | 12 | 1.74 | 1.87 | composed |
| SPACEV, Zipf 0.99 | 1% | Ours, final config | 55.6 | 0.913 | nan | 7.24 | 3.62 | 0.408 | 7.7 | 11.7 | 12 | 2 | 2.25 | composed |
| SPACEV, Zipf 0.99 | 3% | Ours, final config | 44.8 | 0.921 | nan | 7.24 | 4.55 | 0.597 | 7.78 | 12 | 12 | 1.59 | 1.81 | composed |
| SPACEV, Zipf 0.99 | 10% | Ours, final config | 58.1 | 0.915 | nan | 7.24 | 3.6 | 0.595 | 7.79 | 11.9 | 12 | 2.01 | 2.39 | composed |
| SPACEV, Zipf 0.99 | 0.3% | Ours | 44.5 | 0.912 | nan | 7.24 | 4.24 | 0.165 | 7.59 | 11.8 | 12 | 1.7 | 1.8 | composed |
| SPACEV, Zipf 0.99 | 1% | Ours | 47.6 | 0.912 | nan | 7.24 | 4.06 | 0.291 | 7.58 | 11.8 | 12 | 1.78 | 1.91 | composed |
| SPACEV, Zipf 0.99 | 3% | Ours | 52.6 | 0.907 | nan | 7.24 | 3.84 | 0.625 | 7.65 | 11.8 | 12 | 1.88 | 2.11 | composed |
| SPACEV, Zipf 0.99 | 10% | Ours | 57 | 0.907 | nan | 7.24 | 3.82 | 0.894 | 7.77 | 11.8 | 12 | 1.89 | 2.32 | composed |
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
| TriviaQA, Zipf 0.3 | 1% | Ours, final config | 12 | 0.919 | nan | 73.7 | 66.4 | 4.24 | 75.3 | 83.2 | 81.6 | 1.11 | 1.14 | composed |
| TriviaQA, Zipf 0.3 | 10% | Ours, final config | 48.8 | 0.909 | nan | 73.7 | 45.4 | 13.5 | 77.2 | 84.8 | 81.6 | 1.62 | 1.95 | composed |
| TriviaQA, Zipf 0.3 | 1% | Ours | 5.78 | 0.921 | nan | 73.7 | 69.7 | 2.04 | 74.6 | 82.4 | 81.6 | 1.06 | 1.06 | composed |
| TriviaQA, Zipf 0.3 | 10% | Ours | 29.9 | 0.904 | nan | 73.7 | 56.3 | 12.1 | 76.8 | 84.3 | 81.6 | 1.31 | 1.43 | composed |
| TriviaQA, Zipf 0.3 | 1% | Ours (continuous greedy) | 9.28 | 0.919 | nan | 73.7 | 67.8 | 3.28 | 75 | 82.8 | 81.6 | 1.09 | 1.1 | composed |
| TriviaQA, Zipf 0.3 | 10% | Ours (continuous greedy) | 49 | 0.908 | nan | 73.7 | 46.1 | 14.4 | 77.9 | 84.9 | 81.6 | 1.6 | 1.96 | composed |
| TriviaQA, Zipf 0.3 | 1% | LRU | 3.25 | 0.906 | nan | 73.7 | 73 | 4.64 | 75.8 | 83.7 | 81.6 | 1.01 | 1.03 | composed |
| TriviaQA, Zipf 0.3 | 10% | LRU | 53.9 | 0.906 | nan | 73.7 | 44.7 | 15.8 | 79.8 | 86.7 | 81.6 | 1.65 | 2.17 | composed |
| TriviaQA, Zipf 0.3 | 1% | Aker | 2.8 | 0.931 | nan | 73.7 | 71.7 | 0.0313 | 73.8 | 81.7 | 81.6 | 1.03 | 1.03 | composed |
| TriviaQA, Zipf 0.3 | 10% | Aker | 15.3 | 0.958 | nan | 73.7 | 62 | 0.0248 | 73.8 | 81.6 | 81.6 | 1.19 | 1.18 | composed |
| TriviaQA, Zipf 0.3 | 1% | QVCache | 2.21 | 0.831 | nan | 73.7 | 74.7 | 2.19 | 75.9 | 84 | 81.6 | 0.987 | 1.02 | composed |
| TriviaQA, Zipf 0.3 | 10% | QVCache | 15.6 | 0.87 | nan | 73.7 | 63.6 | 0.859 | 74.6 | 82.4 | 81.6 | 1.16 | 1.18 | composed |
| TriviaQA, Zipf 0.6 | 1% | Ours, final config | 23 | 0.916 | nan | 73.5 | 58.7 | 2.51 | 74.5 | 82.3 | 81.9 | 1.25 | 1.3 | composed |
| TriviaQA, Zipf 0.6 | 10% | Ours, final config | 58.6 | 0.909 | nan | 73.5 | 37.3 | 10.6 | 75.3 | 82.4 | 81.9 | 1.97 | 2.42 | composed |
| TriviaQA, Zipf 0.6 | 1% | Ours | 15 | 0.918 | nan | 73.5 | 63.8 | 2.07 | 74.5 | 82.7 | 81.9 | 1.15 | 1.18 | composed |
| TriviaQA, Zipf 0.6 | 10% | Ours | 40.7 | 0.909 | nan | 73.5 | 49.6 | 11.7 | 75.7 | 83.9 | 81.9 | 1.48 | 1.68 | composed |
| TriviaQA, Zipf 0.6 | 1% | Ours (continuous greedy) | 20.8 | 0.919 | nan | 73.5 | 59.5 | 3.5 | 74.9 | 82.8 | 81.9 | 1.23 | 1.26 | composed |
| TriviaQA, Zipf 0.6 | 10% | Ours (continuous greedy) | 59.5 | 0.907 | nan | 73.5 | 38.6 | 13.3 | 76.4 | 83.7 | 81.9 | 1.9 | 2.47 | composed |
| TriviaQA, Zipf 0.6 | 1% | LRU | 14.6 | 0.912 | nan | 73.5 | 64.2 | 4.77 | 75.9 | 84.1 | 81.9 | 1.14 | 1.17 | composed |
| TriviaQA, Zipf 0.6 | 10% | LRU | 61 | 0.908 | nan | 73.5 | 38.7 | 14.5 | 78.5 | 85.6 | 81.9 | 1.9 | 2.57 | composed |
| TriviaQA, Zipf 0.6 | 1% | Aker | 6.53 | 0.944 | nan | 73.5 | 68.5 | 0.0239 | 73.5 | 81.8 | 81.9 | 1.07 | 1.07 | composed |
| TriviaQA, Zipf 0.6 | 10% | Aker | 27.1 | 0.915 | nan | 73.5 | 54.4 | 0.0277 | 73.5 | 81.3 | 81.9 | 1.35 | 1.37 | composed |
| TriviaQA, Zipf 0.6 | 1% | QVCache | 6.5 | 0.922 | nan | 73.5 | 71.5 | 2.03 | 75.6 | 84.3 | 81.9 | 1.03 | 1.07 | composed |
| TriviaQA, Zipf 0.6 | 10% | QVCache | 24.6 | 0.885 | nan | 73.5 | 56.7 | 0.824 | 74.2 | 82.2 | 81.9 | 1.3 | 1.33 | composed |
| TriviaQA, Zipf 0.99 | 1% | Ours, final config | 64.1 | 0.914 | nan | 74.3 | 28.6 | 3.38 | 76.5 | 82.9 | 84.6 | 2.6 | 2.79 | composed |
| TriviaQA, Zipf 0.99 | 10% | Ours, final config | 88.3 | 0.909 | nan | 74.3 | 14.6 | 7.08 | 76.2 | 80.4 | 84.6 | 5.11 | 8.58 | composed |
| TriviaQA, Zipf 0.99 | 1% | Ours | 59.9 | 0.911 | nan | 74.3 | 30.5 | 2.4 | 75.6 | 82.4 | 84.6 | 2.44 | 2.5 | composed |
| TriviaQA, Zipf 0.99 | 10% | Ours | 74.8 | 0.907 | nan | 74.3 | 26.8 | 11.4 | 76.4 | 82.6 | 84.6 | 2.77 | 3.96 | composed |
| TriviaQA, Zipf 0.99 | 1% | Ours (continuous greedy) | 63.4 | 0.911 | nan | 74.3 | 28.7 | 3.61 | 76 | 82.4 | 84.6 | 2.59 | 2.73 | composed |
| TriviaQA, Zipf 0.99 | 10% | Ours (continuous greedy) | 85.8 | 0.908 | nan | 74.3 | 21.1 | 12.2 | 76.8 | 82 | 84.6 | 3.52 | 7.03 | composed |
| TriviaQA, Zipf 0.99 | 1% | LRU | 58.3 | 0.906 | nan | 74.3 | 33.2 | 4.14 | 76.5 | 82.9 | 84.6 | 2.24 | 2.4 | composed |
| TriviaQA, Zipf 0.99 | 10% | LRU | 86.7 | 0.907 | nan | 74.3 | 21.4 | 13.3 | 78.4 | 82.8 | 84.6 | 3.47 | 7.55 | composed |
| TriviaQA, Zipf 0.99 | 1% | Aker | 43.2 | 0.93 | nan | 74.3 | 41.8 | 0.0241 | 74.5 | 82.8 | 84.6 | 1.78 | 1.76 | composed |
| TriviaQA, Zipf 0.99 | 10% | Aker | 59.2 | 0.937 | nan | 74.3 | 29.1 | 0.0205 | 74.5 | 81.6 | 84.6 | 2.55 | 2.45 | composed |
| TriviaQA, Zipf 0.99 | 1% | QVCache | 34.7 | 0.954 | nan | 74.3 | 49.5 | 2 | 76.5 | 85.5 | 84.6 | 1.5 | 1.53 | composed |
| TriviaQA, Zipf 0.99 | 10% | QVCache | 59.7 | 0.916 | nan | 74.3 | 28.8 | 0.384 | 74.8 | 81.7 | 84.6 | 2.59 | 2.48 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | Ours, final config | 52.6 | 0.911 | nan | 11.5 | 5.47 | 0.267 | 11.8 | 14.8 | 15.1 | 2.1 | 2.11 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | Ours, final config | 67.5 | 0.91 | nan | 11.5 | 3.89 | 0.471 | 11.9 | 14.6 | 15.1 | 2.96 | 3.07 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | Ours, final config | 86.2 | 0.908 | nan | 11.5 | 2.1 | 0.681 | 12 | 14.1 | 15.1 | 5.47 | 7.25 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | Ours, final config | 89.2 | 0.907 | nan | 11.5 | 1.85 | 0.582 | 11.9 | 13.7 | 15.1 | 6.2 | 9.29 | composed |
| MedRAG-Zipf (Proximity) | 0.01% | Ours | 42.5 | 0.922 | nan | 11.5 | 6.77 | 0.145 | 11.8 | 14.9 | 15.1 | 1.7 | 1.74 | composed |
| MedRAG-Zipf (Proximity) | 0.03% | Ours | 56.7 | 0.905 | nan | 11.5 | 4.89 | 0.298 | 12 | 14.8 | 15.1 | 2.35 | 2.31 | composed |
| MedRAG-Zipf (Proximity) | 0.1% | Ours | 63.3 | 0.927 | nan | 11.5 | 4.81 | 0.722 | 12.2 | 15 | 15.1 | 2.39 | 2.72 | composed |
| MedRAG-Zipf (Proximity) | 0.3% | Ours | 64.2 | 0.931 | nan | 11.5 | 5 | 0.93 | 12.3 | 15.1 | 15.1 | 2.3 | 2.79 | composed |
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
| TripClick real health-search log | 0.3% | Ours, final config | 9.03 | 0.934 | nan | 9.42 | 8.93 | 0.191 | 9.78 | 14.1 | 13.6 | 1.06 | 1.1 | composed |
| TripClick real health-search log | 1% | Ours, final config | 13.1 | 0.919 | nan | 9.42 | 8.53 | 0.285 | 9.81 | 14 | 13.6 | 1.1 | 1.15 | composed |
| TripClick real health-search log | 3% | Ours, final config | 18.7 | 0.919 | nan | 9.42 | 8.36 | 0.775 | 10.1 | 14.3 | 13.6 | 1.13 | 1.23 | composed |
| TripClick real health-search log | 10% | Ours, final config | 29.2 | 0.913 | nan | 9.42 | 7.65 | 0.894 | 10.3 | 14.5 | 13.6 | 1.23 | 1.41 | composed |
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
| ESCI (no repeated queries) | 0.3% | Ours, final config | 6.34 | 0.925 | nan | 6.73 | 6.53 | 0.215 | 6.9 | 10.1 | 9.87 | 1.03 | 1.07 | composed |
| ESCI (no repeated queries) | 1% | Ours, final config | 8.69 | 0.923 | nan | 6.73 | 6.47 | 0.564 | 6.99 | 10.1 | 9.87 | 1.04 | 1.1 | composed |
| ESCI (no repeated queries) | 3% | Ours, final config | 15.2 | 0.91 | nan | 6.73 | 6.12 | 0.376 | 7.07 | 10.1 | 9.87 | 1.1 | 1.18 | composed |
| ESCI (no repeated queries) | 10% | Ours, final config | 26.7 | 0.909 | nan | 6.73 | 5.61 | 0.792 | 7.26 | 10.4 | 9.87 | 1.2 | 1.36 | composed |
| ESCI (no repeated queries) | 0.3% | Ours | 6.71 | 0.894 | nan | 6.73 | 6.44 | 0.124 | 6.83 | 9.97 | 9.87 | 1.04 | 1.07 | composed |
| ESCI (no repeated queries) | 1% | Ours | 9.16 | 0.905 | nan | 6.73 | 6.34 | 0.295 | 6.89 | 10 | 9.87 | 1.06 | 1.1 | composed |
| ESCI (no repeated queries) | 3% | Ours | 12.6 | 0.899 | nan | 6.73 | 6.22 | 0.261 | 6.97 | 10.1 | 9.87 | 1.08 | 1.14 | composed |
| ESCI (no repeated queries) | 10% | Ours | 21.8 | 0.894 | nan | 6.73 | 5.76 | 0.479 | 7.09 | 10.1 | 9.87 | 1.17 | 1.28 | composed |
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

Sample-rate check (mean ms, cache + DiskANN / DiskANN alone / speedup; whole-run hit rate)

- Zipf 0.6, 1%, Ours: 1/20: 6.72 / 7.11 / 1.059 (h 9.7%, n 1600); 1/10: 6.72 / 7.11 / 1.059 (h 9.7%, n 1600); 1/5: 6.72 / 7.11 / 1.059 (h 9.7%, n 1600)
- Zipf 0.6, 1%, LRU: 1/20: 7.11 / 7.11 / 1.001 (h 9.3%, n 1600); 1/10: 7.11 / 7.11 / 1.001 (h 9.3%, n 1600); 1/5: 7.11 / 7.11 / 1.001 (h 9.3%, n 1600)
- Zipf 0.99, 10%, Ours: 1/20: 3.82 / 7.24 / 1.892 (h 57.0%, n 1598); 1/10: 3.82 / 7.24 / 1.892 (h 57.0%, n 1598); 1/5: 3.82 / 7.24 / 1.892 (h 57.0%, n 1598)
- Zipf 0.99, 10%, LRU: 1/20: 4.01 / 7.24 / 1.804 (h 63.1%, n 1598); 1/10: 4.01 / 7.24 / 1.804 (h 63.1%, n 1598); 1/5: 4.01 / 7.24 / 1.804 (h 63.1%, n 1598)
