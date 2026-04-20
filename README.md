# Basket Mean Reversion via PCA

## Overview
The strategy is developed for a basket of cryptocurrencies which are considered to exhibit similar characteristics and follow common (systematic) trends. It is based on the assumption that any significant deviations from the co-movements of these assets will further lead to their reversion to the common path. Systematic components are constructed through the PCA decomposition, and strategy focuses on the behavior of the remaining part of returns' variance, i.e., residuals. 

The entire strategy is implemented in Jupyter Notebooks. Implementation involves data download, market microstructure modelling, backtesting with hyper-optimization and Walk-Forward Analysis, and performance assessment. The data used comes from Cryptocompare, Binance and ByBit, and is mostly downloaded inside the notebooks through a direct connection to websites or API. The following categories of data are used: spot and futures klines in hourly and minute frequency, trade-level data, funding rates data, fees. All data is obtained from public sources.

The strategy assumes trading takes place on stablecoin-margined futures markets. At the beginning, the universe of 21 memecoins has been selected to develop the strategy. 13 of them have been deemed to have sufficiently liquid futures market to be considered in trading. Out of these 13 assets, 9 are assumed to be traded on Binance, and 4 - on ByBit. The following transaction costs are included in the backtesting: fees, bid-ask spreads, funding rates and price impact.

There are no explicitly given bid-ask spreads in the available datasets. Thus, they must be implied from the trade-level data. The model developed for this purpose assumes there is a fixed daily level of the spread, and the actual spread, observed in the order book, is equal to the sum of this fixed daily spread plus the noise. The fixed daily spread is estimated via linear regression model explaining trade-by-trade price changes with signs of the subsequent orders. The obtained daily spreads are used in the backtesting after smoothing (i.e, the bid-ask spread is constant throughout the day).



## Detailed Description
