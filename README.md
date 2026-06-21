# Movie Recommendation System

CSE 25 Final Project — UC San Diego | Winter Quarter 2026 (January – March 2026)

A movie recommendation engine built with the [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) dataset, implemented as a Kaggle Notebook.

## Overview

This project explores three recommendation approaches:

- **Collaborative Filtering** — recommends movies based on similar users' ratings
- **Content-Based Filtering** — recommends movies based on genre similarity
- **Matrix Factorization (SVD)** — learns latent user and movie factors via SGD to predict ratings

The interaction model used for Matrix Factorization:

```
rating = μ + b_u + b_i + p_u · q_i
```

where `μ` is the global mean, `b_u` and `b_i` are user/item biases, and `p_u · q_i` are learned latent factor vectors.

## Dataset

**MovieLens 20M** — ~20 million ratings from ~138,000 users on ~27,000 movies.

The dataset is loaded directly from Kaggle's input section. An 80/20 per-user train/test split is used for evaluation.

## Evaluation

Model performance is benchmarked against two baselines:
- **Popularity baseline** — always recommends globally highest-rated movies
- **Random baseline** — recommends movies at random (worst-case)

## Tech Stack

- Python (NumPy, pandas, scikit-learn, scikit-surprise)
- Matplotlib / Seaborn for visualization
- Kaggle Notebooks (CPU runtime)
