| workload | size | ours | ours hit | alternative | alt | alt hit | node cache |
|---|---|---|---|---|---|---|---|
| SPACEV, Zipf 0.6 | 1% | 0.992 | 0.181 | aker | 1.04 | 0.0376 | 1.13 |
| SPACEV, Zipf 0.6 | 10% | 1.24 | 0.329 | aker | 1.14 | 0.126 | 1.43 |
| SPACEV, Zipf 0.99 | 10% | 1.94 | 0.581 | aker | 1.8 | 0.449 | 1.4 |
| TriviaQA, Zipf 0.6 | 1% | 1.18 | 0.23 | aker | 1.07 | 0.0653 | - |
| TriviaQA, Zipf 0.6 | 10% | 2.14 | 0.607 | aker | 1.37 | 0.271 | 1.1 |
| TriviaQA, Zipf 0.99 | 10% | 4.52 | 0.883 | qvcache | 2.48 | 0.597 | - |
| MedRAG-Zipf (Proximity) | 0.1% | 3.5 | 0.862 | qvcache | 1.85 | 0.464 | - |
| MedRAG-Zipf (Proximity) | 0.3% | 4.06 | 0.892 | qvcache | 1.86 | 0.464 | 1.05 |
| TripClick real health-search log | 1% | 1.05 | 0.131 | aker | 1.12 | 0.111 | 1.1 |
| TripClick real health-search log | 10% | 1.05 | 0.292 | aker | 1.22 | 0.183 | 1.33 |
| TripClick real health-search log | 30% | 1.51 | 0.477 | aker | 1.31 | 0.242 | 1.82 |
