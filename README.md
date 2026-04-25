# Basket Mean Reversion via PCA


## Overview

The strategy is developed for a basket of cryptocurrencies which are considered to exhibit similar characteristics and follow common (systematic) trends. It is based on the assumption that any significant deviations from the co-movements of these assets will further lead to their reversion to the common path. Systematic components are constructed through the PCA decomposition, and strategy focuses on the behavior of the remaining part of returns' variance, i.e., residuals. 

The entire strategy is implemented in Jupyter Notebooks. Implementation involves data download, market microstructure modelling, backtesting with hyper-optimization and Walk-Forward Analysis, and performance assessment. The data used comes from Cryptocompare, Binance and ByBit, and is mostly downloaded inside the notebooks through a direct connection to websites or API. The following categories of data are used: spot and futures klines in hourly and minute frequency, trade-level data, funding rates data, fees. All data is obtained from public sources.

The strategy assumes trading takes place on stablecoin-margined futures markets. At the beginning, the universe of 21 memecoins has been selected to develop the strategy. 13 of them have been deemed to have sufficiently liquid futures market to be considered for trading. Out of these 13 assets, 9 are assumed to be traded on Binance, and 4 - on ByBit. The following transaction costs are included in the backtesting: fees, bid-ask spreads, funding rates and price impact.

There are no explicitly given bid-ask spreads in the available datasets. Thus, they must be implied from the trade-level data. The model developed for this purpose assumes there is a fixed daily level of the spread, and the actual spread, observed in the order book, is equal to the sum of this fixed daily spread plus the noise. The fixed daily spread is estimated via linear regression model explaining trade-by-trade price changes with signs of the subsequent orders. The obtained daily spreads are used in the backtesting after smoothing (i.e, the bid-ask spread is constant throughout the day).

There are two kinds of price impact models developed for each traded asset. One serves for actual cost calculation (i.e., how a trade actually impacted the price), and allows the price impact to be non-linear in position size. The other one is used in a Markowitz-type (mean-variance) portfolio weight optimization, and it assumes linearity of the price impact, thus ensuring existent, unique, and analytically explicit solution to the optimization problem. Models are inspired by the approach of Almgren et al. (2005). They involve one explanatory variable, which is a product of the realized price volatility in a certain time interval and the position size expressed as a share of trading volume (possibly exponentiated) in a certain time interval. They are estimated via sum of squared errors minimization (Levenberg-Marquardt algorithm and OLS method). The explanatory variable in the model for weights optimization is constructed only using data up to the last whole hour before a given trade. Additionally, observations in this model are weighted in estimation, so that the predicted impact is closer to the non-linear one for smaller positions, which are more likely to be obtained in weights optimization. The models are developed for each asset separately.

Fees are constant, set for each trading pair, and taken from Binance/ByBit websites. The actual funding rates are included in cost calculation if the trade was opened at the moment of a given funding payment. For the purpose of weights optimization, a prediction of the funding rate is included. Predicted funding rate is equal to the current one. However, since the data is only available for funding payment moments, the rates have been interpolated between payments to obtain their values at different times.

The position sizes of the trades (and their direction) are determined in a Markowitz-type (mean-variance) portfolio weight optimization. First, assets are selected for PCA decomposition. Some of them are traded. For each traded asset, a systematic component of the return is obtained using few first PCs. The difference between actual return and systematic component return is a residual return. ARMA models are estimated on the residual time series, and predictions are obtained for each traded asset. They are trained on rolling calibration windows, and their predictions are used on rolling deployment windows. Covariance matrix for the set of traded asset is obtained by combining Pearson correlation matrix (estimated on calibration widow) and EWMA/GARCH volatilities. The price impact has also been incorporated in the optimization formula (for details see [Position Sizing](/tree/main#position-sizing)).


## Detailed Description

dre

wrt3r

### Data

rtr4g

tggt4

tg4

### Position Sizing

sefg

rrr

r

ergteg
