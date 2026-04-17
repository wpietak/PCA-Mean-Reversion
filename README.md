# Basket Mean Reversion via PCA

## Overview
The strategy is developed for a basket of cryptocurrencies which are considered to exhibit similar characteristics and follow common (systematic) trends. It is based on the assumption that any significant deviations from the co-movements of these assets will further lead to their reversion to the common path. Systematic components are constructed through the PCA decomposition, and strategy focuses on the behavior of the remaining part of returns' variance, i.e., residuals. It involves codes and their output for: data download, market microstructure modelling, signal generation and position sizing, hyper-optimization, Walk-Forward Analysis, and out-of-sample performance analysis, along with their descriptions.

## Detailed Description
