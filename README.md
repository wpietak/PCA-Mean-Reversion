# Basket Mean Reversion via PCA


## Overview

The strategy is developed for a basket of cryptocurrencies which are considered to exhibit similar characteristics and follow common (systematic) trends. It is based on the assumption that any significant deviations from the co-movements of these assets will further lead to their reversion to the common path. Systematic components are constructed through the PCA decomposition, and strategy focuses on the behavior of the remaining part of returns' variance, i.e., residuals. 

The entire strategy is implemented in Jupyter Notebooks. Implementation involves data download, market microstructure modelling, backtesting with hyper-optimization and Walk-Forward Analysis, and performance assessment. The data used comes from Cryptocompare, Binance and ByBit, and is mostly downloaded inside the notebooks through a direct connection to websites or API. The following categories of data are used: spot and futures klines in hourly and minute frequency, trade-level data, funding rates data, fees. All data is obtained from public sources.

The strategy assumes trading takes place on stablecoin-margined futures markets. At the beginning, the universe of 21 memecoins has been selected to develop the strategy. 13 of them have been deemed to have sufficiently liquid futures market to be considered for trading. Out of these 13 assets, 9 are assumed to be traded on Binance, and 4 - on ByBit. The following transaction costs are included in the backtesting: fees, bid-ask spreads, funding rates and price impact.

There are no explicitly given bid-ask spreads in the available datasets. Thus, they must be implied from the trade-level data. The model developed for this purpose assumes there is a fixed daily level of the spread, and the actual spread, observed in the order book, is equal to the sum of this fixed daily spread plus the noise. The fixed daily spread is estimated via linear regression model explaining trade-by-trade price changes with signs of the subsequent orders. The obtained daily spreads are used in the backtesting after smoothing (i.e., the bid-ask spread is constant throughout the day).

There are two kinds of price impact models developed for each traded asset. One serves for actual cost calculation (i.e., how a trade actually impacted the price), and allows the price impact to be non-linear in position size. The other one is used in a Markowitz-type (mean-variance) portfolio weight optimization, and it assumes linearity of the price impact, thus ensuring existent, unique, and analytically explicit solution to the optimization problem. Models are inspired by the approach of Almgren et al. (2005). They involve one explanatory variable, which is a product of the realized price volatility in a certain time interval and the position size expressed as a share of trading volume (possibly exponentiated) in a certain time interval. They are estimated via sum of squared errors minimization (Levenberg-Marquardt algorithm and OLS method). The explanatory variable in the model for weights optimization is constructed only using data up to the last whole hour before a given trade. Additionally, observations in this model are weighted in estimation, so that the predicted impact is closer to the non-linear one for smaller positions, which are more likely to be obtained in weights optimization. The models are developed for each asset separately.

Fees are constant, set for each trading pair, and taken from Binance/ByBit websites. The actual funding rates are included in cost calculation if the trade was opened at the moment of a given funding payment. For the purpose of weights optimization, a prediction of the funding rate is included. Predicted funding rate is equal to the current one. However, since the data is only available for funding payment moments, the rates have been interpolated between payments to obtain their values at different times.

The position sizes of the trades (and their direction) are determined in a Markowitz-type (mean-variance) portfolio weight optimization. First, assets are selected for PCA decomposition. Some of them are traded. For each traded asset, a systematic component of the return is obtained using few first PCs. The difference between actual return and systematic component return is a residual return. ARMA models are estimated on the residual time series, and predictions are obtained for each traded asset. They are trained on rolling calibration windows, and their predictions are used on rolling deployment windows. Covariance matrix for the set of traded asset is obtained by combining Pearson correlation matrix (estimated on calibration widow) and EWMA/GARCH volatilities. The price impact has also been incorporated in the optimization formula (for details see [Signal Generation and Position Sizing](/README.md#signal-generation-and-position-sizing)).

The assets to be used in the PCA decomposition, traded assets, as well as the number of PCs to construct systematic components and ARMA orders for each traded asset are selected in the hyper-optimization (hyper-tuning) in Walk-Forward Analysis (WFA). First, a genetic optimization algorithm is run on the in-sample window to select the assets for PCA decomposition. A simplified trading procedure and return calculation is assumed with aim to find stable performance plateau by removing best returns, smoothing performance (Sharpe) and adding penalties for over-reliance on single assets. Next, trading candidates (assets, their number of PCs and ARMA orders) are selected by checking performance in individual trading, and a prioritized search method is used to select the final combinations. Similarly, the best returns are removed, performance (Sharpe) is smoothed, and penalties for over-reliance on single assets are added. The strategy is then deployed on the out-of-sample window, and the windows are rolled until the end of the backtesting period. Out-of-sample performance analysis is provided.


## Detailed Description

Details of the strategy methodology and backtesting procedure are provided below.

### The Idea

The strategy has been built for a basket of cryptocurrencies that are believed to follow the same (short-term) trends, or react to the macro events in a similar manner. Their systematic components are assumed to be the main drivers of their direction, and any deviations are usually an effect of a temporary non-adjustments that will be corrected soon. Ideally, a basket for a mean reversion via PCA strategy would be composed of assets which share a very similar sentiment among market participants, react in a very similar way to events/information regarding broad market (in terms of direction and intensity), and very rarely follow sustained individual trends driven by asset-specific factors (fundamental or any other).

For such a basket of assets, PCA transformation will be performed in order to construct systematic components of the selected assets' returns by projecting one or few first Principal Components into the original space. These systematic components are meant to reflect parts of the returns which can be attributed to the overall short-term trend of the class of assets represented by the basket. The remaining parts of the returns (residuals) are assumed to exhibit mean reversion, such that the total returns eventually converge to the overall trend. Hence, the time series of the residuals are used in modelling, and the obtained predictions are assumed to be the predictions of the total returns, since the systematic components are orthogonal to the residuals.

The initial universe of cryptocurrencies considered in the strategy is composed of 21 memecoins which have been created around 2023-24, and maintained relative attention until mid-2025. They are deemed to possess the required characteristics due to the following reasons:
- they all emerged during the same cycle, mostly within the same stage,
- all of them, at least at some point, grew large enough, and gained enough attention to significantly strengthen their survivability (Lindy effect), but never came close to majors (like, e.g., DOGE), thus obtaining a specific positioning,
- they share the same risk sentiment,
- they are recognized and traded mostly by similar market participants,
- their value does not depend on any fundamental factors, and it is not strictly attached to specific figures, companies, organizations or events which could play a decisive role in their market perception,
- for the vast majority of time, their price action is not idiosyncratically driven by news, announcements or events; they usually react to external factors in a similar manner.

The names/shortcuts/tickers of memecoins belonging to the initial universe, as well as those selected as candidates for trading, can be found in XXX notebook.

### Data

The following types of data are used in the backtesting:
- Klines (OHLC and volume data)
  - Cryptocompare - hourly data for spot USDT pairs (obtained directly or through conversion) from different CEXs for all 21 memes, downloaded directly through cryptocompare library
  - Binance - hourly and minute data for perpetual futures USDT or USDC pairs (USDT/C-margined) for 9 selected memes, downloaded directly via link to data.binance.vision website
  - ByBit - hourly and minute data for perpetual futures USDT pairs (USDT-margined) for 4 selected memes, obtained by transforming trade-level data downloaded directly via link to public.bybit.com website
- Trade-level data (price, volume, timestamp, taker side)
  - Binance - transactions aggregated on market order & price level (each record tells how much volume was traded for a given price within a given market order) for the same pairs as klines, downloaded directly via link to data.binance.vision website
  - ByBit - transactions aggregated on market order & limit order level (each record tells how much volume was traded within a given market order against a given limit order) for the same pairs as klines, downloaded directly via link to public.bybit.com website
- Funding rates (payment time and rate)
  - Binance - indication of payment times and corresponding funding rates for the same pairs as klines, downloaded directly via link to data.binance.vision website
  - ByBit - indication of payment times and corresponding funding rates for the same pairs as klines, downloaded in csv from bybit.com, and imported in the notebook
- Fees - rates for (USDT or USDC) perpetual futures from fee tables on binance.com and bybit.com

Klines from Cryptocompare are only used for signal generation, and more specifically - for PCA transformation. Hourly klines for Binance and ByBit futures are used for signal generation / position sizing, including PCA transformation, residual time series construction, and volatility and correlation calculation, as well as trading costs and returns calculation. Minute klines are used to estimate realized volatility, which is used in price impact model. Spot price data from Cryptocompare for 13 assets selected for trading is not used in the presented backtesting.

Trade-level data has been used to construct bid-ask spread time series and develop price impact models. As for the bid-ask spread time series, daily spreads for each asset were estimated one by one by downloading the trade-level data for each day and calculating the spread with several candidate models. Depending on the model, trade records might have been aggregated on timestamp & taker side level in order to group transactions belonging to the same order. Such aggregation was also performed to prepare the datasets used to estimate price impact models. The latter datasets involve all orders filled at more than one price with average realized price impact recorded for ~7 first months of the backtesting period. The bid-ask spreads time series are constructed for the entire backtesting period.

Funding rate data does not require transformation to calculate realized cost. However, linear interpolation between funding payment times was performed to obtain funding rate predictions for each full hour, since the rate levels are only available for payment times. These predictions are used for signal generation.

### Bid-Ask Spread

Construction of the bid-ask spread time series is composed of the following steps:
- For each of the 13 assets selected for trading:
  - for each day within the backtesting period (plus some buffer - this is required for smoothing):
    - download trade-level data,
    - estimate model parameters, and derive bid-ask spread based on them (there were 8 different models considered),
    - save estimated spreads and selected parameters;
  - imply minimum tick size for each day from klines.
- Plot generated time series, obtain statistics, analyze the results, and select the final model.
- Floor the bid-ask spread for each day with the tick size, and smooth the time series (after flooring).

The following approaches have been considered to estimate daily bid-ask spreads:
- <b>(1) Delta-based:</b> Stoll (1989) models price increments as $\Delta P_t = Q_t \delta S$ if $Q_t = Q_{t-1}$, and $\Delta P_t = Q_t (1 - \delta) S$ if $Q_t <> Q_{t-1}$, where $Q_t$ is a buy/sell indicator, $S$ is the bid-ask spread, and $\delta$ is a parameter. This approach estimates $\delta S$ and $(1 - \delta) S$ by calculating mean price changes for continuation ($Q_t = Q_{t-1}$) and reversal ($Q_t <> Q_{t-1}$), respectively, and infers the bid-ask spread from them.
- <b>(2) Lambda-based:</b> Huang and Stoll (1997) model price increments as $\Delta P_t = \frac{S}{2} \Delta Q_t + \lambda \frac{S}{2} Q_{t-1} + e_t$, where $\lambda$ is a parameter reflecting the impact of private information reveal, and $e_t$ is a stochastic term capturing public information reveal and rounding error. In this approach a two-step model is developed. First, $\lambda \frac{S}{2}$ is estimated by regressing $\Delta P_t$ on $Q_{t-1}$ using continuation trades observations only. Next, obtained coefficient is used to estimate $\frac{S}{2}$ by regressing the estimation of $\Delta P_t - \lambda \frac{S}{2} Q_{t-1}$ on $\Delta Q_t$ using all observations.
- <b>(3a) Covariance-based with capped lambda:</b> This is an enhanced approach of Roll (1984), which uses Huang and Stoll (1997) model of price increments, but focuses on reversal trades only. We consider $E \left[\Delta P_{t_{r}} \Delta P_{t_{pr}} \right] = E \left[\frac{S^2}{4} \left(\Delta Q_{t_r} + \lambda Q_{t_{r}-1} \right) \left(\Delta Q_{t_{pr}} + \lambda Q_{t_{pr}-1} \right) \right] = - \frac{S^2}{4} (2 - \lambda)^2$, where $t_r$ iterates over reversal trades, and $t_pr$ is timestamp of the previous reversal trade before $t_r$, assuming $e_t$ is not autocorrelated. Then, $S = \frac{2 \times \sqrt{- cov \left(\Delta P_{t_r}, \Delta P_{t_{pr}} \right)}}{2 - \lambda}$, assuming $E[\Delta P_t] = 0$. In order to estimate the spread, reversal trades are filtered to obtain time series of first transactions price changes of alternate buy and sell orders, and their autocovariance is calculated. Next, the autocovariance estimator is inserted into the above formula, together with $\lambda$ estimator obtained for the purpose of Lambda-based approach (2). Here, $\lambda$ is capped at 1.
- <b>(3b) Covariance-based with lambda:</b> This approach is the same as (3a), except there is no cap on $\lambda$ here.
- <b>(4a) Market impact with side indicator linear regression:</b> This approach is based on a model of differences between the last prices of subsequent orders, which has the following formula: $\Delta P_t = \beta_1 Q_t + \beta_2 Q_t V_t + e_t$, where $\beta_1$ and $\beta_2$ are coefficients, and $V_t$ is the volume of a given order, expressed as a number of contracts traded. $\beta_1$ is supposed to reflect a fixed impact of the half-spread, while $\beta_2$ is supposed to reflect a variable impact, depending on order size. In order to estimate the model, trades are grouped on timestamp & taker side level, and last price and volume sum are taken for each record (order). Then, linear regression is performed, with taker side and product of taker side and volume being the explanatory variables, and price difference being the dependent variable. The first coefficient is multiplied by 2 to obtain the bid-ask spread estimation. 
- <b>(4b) Market impact with side indicator lag linear regression:</b> This is an enhancement of the approach (4a), based on a model with the following formula: $\Delta P_t = \beta_1 \Delta Q_t + \beta_2 Q_{t-1} + \beta_3 Q_t V_t + e_t$. Thus, the assumptions regarding the fixed impact are aligned with Huang and Stoll (1997) model. The estimation procedure is analogous as in approach (4a), and, similarly, the bid-ask spread estimation is obtained by multiplying the first coefficient by 2.
- <b>(5a) Huang-Stoll-based linear regression 1:</b> This approach is based on Huang and Stoll (1997) model: $\Delta P_t = \frac{S}{2} \Delta Q_t + \lambda \frac{S}{2} Q_{t-1} + e_t$, but focusing on differences between last and first prices of subsequent orders only. That is, only the price increments between the last transaction of a given order and the first transaction of the next order are modelled, while price differences between transactions belonging to the same order are not considered. For the purpose of estimation, trades are grouped on timestamp & taker side level, and first and last price are taken for each record (order). Next, for each order, the difference between its' first price and previous order's last price are calculated. Then, a linear regression model, involving $\Delta Q_t$ and $Q_{t-1}$ as explanatory variables and obtained price difference as dependent variable, is estimated via OLS method. First coefficient multiplied by 2 serves as an estimation of the bid-ask spread. Huang and Stoll (1997) use Generalized Method of Moments (GMM) as their estimation method, because they also consider models with additional moment conditions imposed, resulting in their overidentification. However, only the basic version - in which moment conditions involve orthogonality of residuals and explanatory variables solely - is considered here. In such case GMM is equivalent to OLS, since the number of parameters is then equal to the number of moment conditions (exact identification), and thus, there exists a unique set of parameters for which the vector of moment conditions is equal to 0.
- <b>(5b) Huang-Stoll-based linear regression 2:</b> Huang and Stoll (1997) propose an extension of the model presented above, in which they allow for non-zero serial correlation in trade/order flow. Denoting reversal probability as $\pi$, we get $E \left[ Q_{t-1} | Q_{t-2} \right] = (1 - 2 \pi) Q_{t-2}$. Thanks to this, it is also possible to distinguish adverse selection ($\alpha$) and inventory risk ($\beta$) components of private information reveal impact ($\lambda = \alpha + \beta$). This is due to the fact that quote adjustments for inventory reasons depend only on actual trades, while adjustments for adverse selection reasons can also take into account order sign expectations. Hence, only the adverse selection component of quote adjustment requires correction for expectation, and, consequently, in the model there appears a term dependent on $\alpha$ which does not have a $\beta$ analog. The resulting final formula for price changes is the following: $\Delta P_t = \frac{S}{2} Q_t + (\alpha + \beta - 1) \frac{S}{2} Q_{t-1} + \alpha \frac{S}{2} (1 - 2 \pi) Q_{t-2} + e_t$. Note that it involves 4 parameters and only 3 explanatory variables. Huang and Stoll (1997) overcome this by using the bid-ask spreads observed for particular trades. However, $\pi$ also appears in the formula provided earlier, and it can be estimated separately. Here, we use the same dataset as in approach (5a), and we estimate $\pi$ as a frequency of order reversals. This estimator is then used to construct the third explanatory variable in the linear regression model based on the above formula, esitmated via OLS, similarly as in case of approach (5a). Analogously, the first obtained coefficient multiplied by 2 serves as an estimation of the bid-ask spread.

The implementation of the entire bid-ask spread time series construction process, including data download and preprocessing, models estimation, and results summary and analysis, is presented in XXX notebook. The eventually selected approach is <b>(5a) Huang-Stoll-based linear regression 1</b>. Flooring and smoothing was applied to the spreads estimated with this approach to obtain the final time series used in backtesting.

Floor is applied to each daily spread at the tick size level. Tick size is implied from klines based on the maximum number of digits after decimal point in the quoted prices for a given day. In order to reduce the impact of short-term noise on estimations, smoothing is applied to the floored spreads by taking a 5-day centered moving average weighted with a triangular kernel. 

Constant daily bid-ask spreads are meant to reflect average market conditions prevailing in particular days. In reality, the observed spreads differ from the model constant spread by a stochastic term. Its' impact is supposed to average out across time.

### Price Impact Model

There are two price impact models developed for the purpose of backtesting for each asset selected for trading - Cost Calculation (CC) Model and Weights Optimization (WO) Model. Cost Calculation Model is used to calculate the actual impact which given trade had (or would have) on the asset price/return. Weights Optimization Model is used to provide parameters which would enter the formula assigning optimal portfolio weights (see [Signal Generation and Position Sizing](/README.md#signal-generation-and-position-sizing)); in other words, it provides predictions used by other model to decide on position sizes. 

It is assumed that the price impact, caused by taken trades, fully decreases the return earned. This can be interpreted as price impact being fully temporary, or it being permanent (at least within the trading window), but entirely eliminating buyers/sellers (depending on trade side) that would otherwise push price by the same amount (i.e., they are unwilling to make the same trades at more adverse prices).

The models have been estimated using trade-level data aggregated on timestamp & taker side level with volume-weighted average price and resulting impact calculated for each record (order). Additionally, realized volatility, estimated using minute klines, and trading volume, taken from hourly klines, were used to transform the explanatory variable. The models are estimated once, using time series from 22.01.2025 to 16.08.2025 (usually millions of observations), in order to facilitate model development, choice, testing, analysis and usage. They are intended to reflect a sufficiently robust form of relationship between a market order dollar volume and price impact.

#### Cost Calculation Model

The purpose of the Cost Calculation Model is to provide price impact predictions which would be as close to the impacts expected to be observed in reality (for given conditions) as possible. It is inspired by Almgren et al. (2005) approach to price impact modelling. The aim of Almgren et al. (2005), however, was to distinguish permanent and temporary impact for large orders, usually splitted into several child orders. Hence, the functional form of their model involves two equations:

$$\frac{I}{\sigma} = \gamma T sgn(X) \left| \frac{X}{VT} \right|^{\alpha} \left( \frac{\Theta}{V} \right)^{\delta} + \left< noise \right>$$

$$\frac{1}{\sigma} \left( J - \frac{I}{2} \right) = \eta sgn(X) \left| \frac{X}{VT} \right|^{\beta} + \left< noise \right>,$$

where:
- $I$ - permanent impact, i.e., relative price change between order execution start and finish plus some buffer,
- $J$ - realized impact, i.e., relative difference between price at order execution start and average obtained price,
- $X$ - order size in number of shares with respective sign (positive for a buy and negative for a sell),
- $V$ - moving average of given stock's daily volume in number of shares,
- $T$ - volume time of execution, i.e., fraction of an average day's volume realized between start and finish of order execution,
- $\sigma$ - realized daily volatility,
- $\Theta$ - the total number of shares outstanding,
- $\gamma, \alpha, \delta, \eta, \beta$ - coefficients to be estimated.

However, we are only concerned about the realized impact of a single (child) order. Thus, we only need one formula. Moreover, time execution duration and outstanding shares would not even be applicable (especially given that we develop a separate model for each asset). Therefore, our model formula is the following:

$$J = \eta \sigma sgn(X) \left| \frac{X}{V} \right|^{\beta} + \left< noise \right>,$$

where:
- $J$ - realized impact, i.e., relative difference between first and average obtained price,
- $X$ - order volume in dollars (USDC/USDT) with respective sign (positive for a buy and negative for a sell),
- $V$ - moving average of given asset's daily/hourly volume in dollars (USDC/USDT),
- $\sigma$ - moving average of given asset's realized daily/hourly volatility,
- $\eta, \beta$ - coefficients to be estimated.

There are a few additional modifications introduced. Firstly, realized volatility scales the explanatory variable, and not the dependent variable, so that the model directly predicts the price impact for a given order. Secondly, volumes are expressed in dollars (USDC/USDT), because the Position Sizing Model will provide fractions of portfolio's value to be allocated into particular assets. On top of that, there were several different variants of assets' volume and realized volatility tested.

Daily (hourly) volumes are taken from klines, and they are calculated as a sum of traded contracts quantities multipled by their corresponding prices within a given day (hour). Daily (hourly) realized volatility is a square root of daily (hourly) realized variance, which is calculated as a sum of minute log-returns squares within a given day (hour). Minute returns are obtained using minute klines. In the basic model version, simple daily volume and daily relized volatility are used (i.e., moving average is not applied). Further, models with the following moving average types are considered:
- centered moving average of daily realized volatility and daily volume with triangular kernel weighting,
- centered moving average of hourly realized volatility and hourly volume with Gaussian kernel weighting,
- exponentially-weighted moving average (EWMA) of daily realized volatility and daily volume,
- exponentially-weighted moving average (EWMA) of hourly realized volatility and hourly volume.

The models are estimated through the least squares method, using Levenberg-Marquardt algorithm, which is a combination of Gauss-Markov algorithm (used by Almgren et al. (2005)) and gradient descent. Usage of such algorithm is necessary to estimate the value of exponent ($\beta$ in the formula above). However, once the exponents are obtained, standard OLS is applied, and linear coefficients ($\eta$) estimated with both methods coincide. For the estimated linear regression models, the summaries of standard statistics ($R^2$ in particular) are analyzed. Eventually, the model with <b>EWMA of hourly realized volatility and hourly volume</b> is selected to be used in the backtesting.

Additionally, the following typical linear regression tests have been performed:
- ADF and KPSS tests for stationarity of the dependent variable and residuals,
- Lagrange Multiplier test for autocorrelation of residuals,
- Breusch-Pagan test for heteroskedasticity,
- RESET for correctness of specification,
- Kolmogorov-Smirnov test for normality of error term distribution.

Moreover, various types of impact analysis have been performed for particular assets. This includes:
- one-time price impact for different order sizes in average conditions,
- impact distributions for different order sizes,
- full costs of trade open and close (fees, bid-ask spread, funding and impact) for different order sizes in average conditions,
- full costs of trade open and close for different order sizes in different conditions,
- evolution of costs over time.

Data download and preprocessing, and model estimation, analysis and testing can be found in XXX notebook.

#### Weights Optimization Model

The purpose of the Weights Optimization Model is to provide price impact predictions which will be used to decide about the optimal position size. More specifically, it provides a parameter of a linear function describing relationship between the position size and resulting price impact. This function is involved in the formula of the objective function - which is a subject of optimization with respect to portfolio weights - and it must be linear for the solution to be found analytically (for details see [Signal Generation and Position Sizing](/README.md#signal-generation-and-position-sizing)).

It is thus meant to linearly approximate the actual impact - given by the Cost Calculation Model - from an ex ante perspective, that is, before a given order is placed. Hence, asset's volume and realized volatility up to the last full hour before a given order is taken into account. Therefore, the model formula is the following:

$$J_i = \eta_{lin} \times \sigma_{t(i)-1} \times \frac{X_i}{V_{t(i)-1}} + e_i,$$

where:
- $J_i$ - realized impact of order $i$, i.e., relative difference between first and average obtained price,
- $X_i$ - order $i$'s volume in dollars (USDC/USDT) with respective sign (positive for a buy and negative for a sell),
- $t(i)$ - an hourly interval within which order $i$ has been filled (e.g., 4 PM - 5 PM for an order filled at 4:17:33.258 PM),
- $V_t$ - EWMA of given asset's hourly volume in dollars (USDC/USDT) up to hour $t$ included,
- $\sigma_t$ - EWMA of given asset's realized hourly volatility up to hour $t$ included,
- $e_i$ - error term,
- $\eta_{lin}$ - coefficient to be estimated.

The model is developed only for the moving averages corresponding to those in the selected Cost Calculation Model, and the same dataset is used for model estimation. The model is estimated with Weighted Least Squares (WLS) method. Weighting is adjusted such that the linear approximation is closer to the actual impact for order volumes that will be most often observed in trading simulation during backtesting. Observations with order volumes within ranges expected to be usually given by the Position Sizing Model (based on returns predictions) are assigned higher weights. 

More specifically, weights are assigned based on Cost Calculation Model impact predictions for particular observations, using Gaussian kernel with specified impact center. Base model with equal weighting and models with different kernel center and scaling parameters are developed for each asset. Predictions for a non-linear model are compared visually with predictions for various linear models, and a kernel center & scaling pair is selected for each asset to construct weights. The final models are estimated, and WLS summaries and impact comparisons for WO vs. CC Models are provided. The estimation procedure and all analyses are presented in XXX notebook. 

### Signal Generation and Position Sizing

The procedure determining positions to take is the following:
1. PCA transformation and inverse transformation are performed to construct systematic components and resulting residuals (difference between return and its' systematic component). The projection matrix is constructed on a rolling calibration window, and it is further used to perform (inverse) transformation on a rolling deployment window.
2. Using constructed residual time series, ARMA is trained on the rolling calibration window and predictions are generated for the rolling deployment window. Residuals predictions are used as returns predictions in further steps, since residual, by construction, is orthogonal to the systematic component (i.e., remaining part of the return).
3. A trade is taken in a given period for a given asset, if its' predicted return exceeds by some buffer the fixed costs of opening and closing a trade on a given side (long if prediction is positive and short if it is negative) in this period. The fixed costs include fees (paid twice), bid-ask spread (two half-spreads) and funding rate (or funding rate prediction if funding is not paid at the open), while buffer is a defined parameter.
4. The final position sizes are determined by the analytical solution to a Markowitz-type (mean-variance) optimization formula with price impact incorporated.

The strategy assumes a fixed time horizon (in hours) is set for predictions and trading. Only one-time-step predictions are obtained, thus all return time series must have frequency aligned to this horizon. Positions are also held for this horizon only, unless the same signal is generated in the next time-step. In such case position can be increased or partially reduced. In the presented backtesting only 1-hour horizon is considered.

#### PCA (Inverse) Transformation

For a selected basket of assets, PCA transformation is performed on the calibration window, and returns of $k$ first Principal Components (PCs) are obtained:

$$Z_{n \times k}^{cal} = \left( X_{n \times d}^{cal} - J_{n \times d} \mu_{d \times 1}^{cal} \right) W_{d \times k},$$

where:
- $Z^{cal}$ - matrix of PCs' returns in the calibration window,
- $X^{cal}$ - matrix of assets' returns in the calibration window,
- $J$ - all-ones matrix,
- $\mu^{cal}$ - vector of assets' average returns in the calibration window,
- $W$ - projection matrix,
- $n$ - number of observations in the calibration window,
- $d$ - number of assets in the basket.

Next, inverse transformation is performed to obtain the systematic components of assets' returns (i.e., PCs projection into the original space) and resulting residuals:

$$S_{n \times d}^{cal} = Z_{n \times k}^{cal} W_{k \times d}^{\top} + J_{n \times d} \mu_{d \times 1}^{cal}$$
$$R_{n \times d}^{cal} = X_{n \times d}^{cal} - S_{n \times d}^{cal},$$

where:
- $S^{cal}$ - matrix of assets' returns' systematic components in the calibration window,
- $R^{cal}$ - matrix of assets' returns' residuals in the calibration window.

The data from calibration window is mean-centered for the purpose of transformation. In result, systematic component vectors have the same means as total return vectors, and residual vectors have means equal to 0. The latter is desired, since ARMA models will be trained on residual time series. However, mean-centering is not applied to deployment window data, because unconditional expected values of the returns are assumed to be 0, while mean-centering would imply estimating them based on calibration period sample averages, which are not considered to be accurate estimators. Therefore, systematic components and residuals in the deployment period are obtained as follows:

$$S_{m \times d}^{dep} = X_{m \times d}^{dep} W_{d \times k} W_{k \times d}^{\top}$$
$$R_{m \times d}^{dep} = X_{m \times d}^{dep} - S_{m \times d}^{dep},$$

where:
- $X^{dep}$ - matrix of assets' returns in the deployment window,
- $m$ - number of observations in the deployment window,
- $S^{dep}$ - matrix of assets' returns' systematic components in the deployment window,
- $R^{dep}$ - matrix of assets' returns' residuals in the deployment window.

Residual time series of selected assets are used in the next steps. Only 13 assets selected for trading can be considered, but, depending on the step in WFA process, the subset may be further reduced. Additionally, different values of $k$ may be chosen for different assets. Hence, the transformation is performed for a range of $k$ values.

#### ARMA

For selected assets $ARMA(p, q)$ model is estimated on calibration period residual time series (i.e., selected columns of $R^{cal}$ are used to train ARMA). 10 different combinations of $p$ and $q$ have been considered in the backtesting, including all combinations for $p, q = 0, 1, 2$ (except for $(0, 0)$) as well as $(3, 0)$ and $(0, 3)$. Usually, models were estimated with state space model estimation method (except for the initial stage of hyper-optimization, where only $AR(2)$ is applied to select baskets for PCA).

Once ARMA parameters are estimated, one-time-step predictions are generated for deployment period residual time series (i.e., selected columns of $R^{dep}$) without model refitting. If the process generated by the model is too persistent, implying lack of visible mean reversion (i.e., $p$ is too large), predictions will be overwritten with 0. These predictions are further used to produce signals and determine position sizes.

#### Trade Signal

For each of the selected assets, a trade will be taken in a given time-step, if the absolute value of predicted residual exceeds the corresponding fixed cost by a specified buffer or more. The following fixed cost components are considered:
- <b>Fees:</b> 2 times the lowest tier taker trading fee (for open and close) specified by a given exchange for a given pair. It is assumed fees are known <i>ex ante</i>, since they are constant and explicitly given. They are already expressed in the required form, i.e., in terms of position value's fraction.
- <b>Bid-Ask Spread:</b> 2 times the half-spread (for open and close), i.e., a full spread, estimated for a given day, as described in the [Bid-Ask Spread](/README.md#bid-ask-spread) section. Bid-ask spreads are expressed as a fraction of given asset's price, so that they are comparable with return (residual) prediction. Spreads themselves are assumed to be known <i>ex ante</i>, even at trade close (which is necessary given their construction), but the price at trade close is unknown <i>ex ante</i>. Thus, when the fixed cost is calculated for the purpose of signal generation and position sizing, the half-spread is expressed as a fraction of the previous bar's close price twice. For the purpose of calculation of the actual costs incurred, prices at the previous bar's close and the current bar's close are taken for a trade open and close, respectively. Note that the prices provided in klines are actual transaction prices, and not mid-prices. However, they serve as the best approximation of the actual mid-prices.
- <b>Funding Rate:</b> Funding rate paid at trade open plus predicted funding rate paid at trade close. Funding is supposed to be exchanged at full hours between the sides of the positions opened at that time only. However, in practice a certain interval after the full hour is specified, and the sides of all positions opened at any point during this interval are required to exchange the funding. Thus, it is assumed that fundung must be paid, even if the position is opened shortly after the full hour. Funding rate at trade open is known <i>ex ante</i>, while the rate at trade close is predicted to be the same as the current rate at trade open. Since only the rates at funding payment times are available, interpolation is performed to obtain rates for the remaining times. Analogously as in the case of bid-ask spread, for the purpose of actual incurred costs calculation, the actual funding rate at trade close is taken. Additionally, since funding to be paid (or received) depends on trade side, the rates are multiplied by the sign of the corresponding residual prediction to obtain the cost (which may be negative).

The net predicted residuals, i.e., predicted residuals less the corresponding fixed costs, are further used to determine the position sizes. 

#### Markowitz-type Optimization

In order to determine the final position sizes, a Markowitz-type optimization problem with price impact is solved. In the classical formulation of the problem, we have an agent maximizing expected utility of CARA type ($u(x) = 1 - e^{-\lambda x}$, where $\lambda$ is a risk-aversion parameter). If $x$ is the return on a portfolio of $d$ assets whose returns are normally distributed with expected value vector $\mu$ and covariance matrix $\Sigma$, then the agent chooses weights $w$ such that $w^{\top} \mu - \frac{\lambda}{2} w^{\top} \Sigma w$ is maximized, typically with a constraint that $\sum_{i=1}^{d} w_i = 1$.

In our framework, the constant vector of assets' returns' expected values $\mu$ is replaced with a vector-valued function of weights $\mu(w)$. The dependence of expected values on weights is due to the existence of price impact. The price impact is assumed to be deterministic, and hence, $\mu(w)$ is not stochastic. Additionally, there is no constraint on weights imposed, since, on one hand, it is not required to allocate the entire portfolio into the traded assets, and, on the other hand, leverage may be used. (However, in practice the model generates weights which sum is always below 1.) Then, the optimization problem becomes:

$$\max_w f(w) = w^{\top} \mu(w) - \frac{\lambda}{2} w^{\top} \Sigma w$$

If we assume that the price impact is linear, and the expected value of a single asset's return is simply net predicted return (in our case - residual) minus the order volume multiplied by a respective parameter, that is:

$$\mu(w) = s - G w,$$

where:
- $s$ - a vector of net predicted returns (here: predicted residuals less the corresponding fixed costs),
- $G$ - a diagonal matrix of linear impact parameters; in our case: $G = diag{\left( \left[ g_1 ... g_d \right] \right)}$,
- $g_i = 2 \times \eta_{lin, i} \times \frac{\sigma_i}{V_i} \times p$ for $i = 1, ..., d$, where $\eta_{lin, i}$ is the estimated coefficient, $\sigma_i$ and $V_i$ are respective EWMA realized volatility and EWMA volume (as defined for [Weights Optimization Model](/README.md#weights-optimization-model)) for asset $i$, and $p$ is the current portfolio value,

then we have:

$$f(w) = w^{\top} \left( s - G w \right) - \frac{\lambda}{2} w^{\top} \Sigma w = w^{\top} s - w^{\top} \left( G + \frac{\lambda}{2} \Sigma \right) w,$$

and:

$$\nabla f(w) = s - \left( 2 G + \lambda \Sigma \right) w$$

Since $2 G + \lambda \Sigma$ is positive definite (see below), Hessian of $f$ is negative definite, and thus, $\nabla f(w) = 0$ gives a local (and in this case - global) maximum. Therefore, the solution to the optimization problem defined above is:

$$w^* = \left( 2 G + \lambda \Sigma \right)^{-1} s$$

It can be easily shown that $2 G + \lambda \Sigma$ is strictly positive definite, and hence invertible. If $\Sigma$ is a proper covariance matrix (without perfect correlations), and $\lambda$ is strictly positive (which it is by definition), then $\lambda \Sigma$ must be positive definite. $G$ (and thus $2G$) is trivially positive definite, given it is a diagonal matrix with positive entries (price impact is positive for a buy and negative for a sell). Then, $w^{\top} \left( \lambda \Sigma \right) w > 0$, $w^{\top} \left( 2 G \right) w > 0$, and $w^{\top} \left( 2 G + \lambda \Sigma \right) w > 0$ for all $w \neq 0$.

$\Sigma$ is constructed by combining empirical correlation matrix estimated on the respective calibration period (the same as in case of PCA and ARMA) and rolling volatilities re-estimated each time-step. In the presented backtesting only EWMA volatility has been used. $\lambda$ is a parameter which may be adjusted according to needs.

The portfolio sizing optimization is solved in every time-step separately, while PCA and ARMA are recalibrated and deployed periodically on rolling calibration and deployment windows, respectively.

### Trading and Return Calculation

The full procedure of trading simulation and returns (P&L) calculation, applied in out-of-sample period and at late stages of hyper-optimization over in-sample period, is the following (for any single time-step $t$ in the trading period):
1. If $t$ is not the first time-step, check if any positions from $t-1$ need to be kept (i.e., if there is a signal of same sign in $t$ as in $t-1$). For the positions that are kept, take values from before close in $t-1$, and for the rest - from after close in $t-1$ (step 6.). Use them to calculate returns on particular assets and P&L for $t-1$.
2. Take correlation matrix estimated on the corresponding calibration period and volatilities estimated for $t$, and construct the covariance matrix. Next, obtain optimal portfolio weights $w^*$ for $t$, using the derived formula. A cap on leverage is applied. If the sum of absolute weights exceeds a specified threshold, weights are proportionately scaled down.
3. Based on $w^*$, target position values for $t$ are calculated and compared with outstanding positions (not closed due to signal persistence) to determine adjustments necessary to make.
4. For any position that is partially closed, the price impact of the reduction is calculated using [Cost Calculation Model](/README.md#cost-calculation-model). Together with respective fixed costs (fees, half-spread and funding) it constitutes total costs of the reduction, which will be included in the realized P&L for $t$. For the remaining positions requiring adjustments (opening/increasing), the price impact and fixed cost at the open are calculated in an analogous way. Fixed cost is deducted from the target position value to obtain the position value after opening (net of opening costs).
5. The engine allows to set a stop-loss (SL) specified in terms of a fraction of portfolio value lost on a single position in a given time-step. If, for either high or low price of a given asset in a given time-step (taken from klines), the portfolio return on a position is such that the SL would be triggered, the final return on this position is equal to the SL condition's return. Otherwise, it is equal to the standard return ($t$ close vs. $t-1$ close). However, in the presented backtesting SL was set for -100% return, thus it effectively had no use.
6. The position value after opening is decreased by the price impact at the open and increased by the final return from step 5. to obtain the position value before close. Position value before close is decreased by the price impact (of closing the entire position) and fixed cost at the close (both calculated analogously as at the open) to obtain the position value after close.
7. If $t$ is the last time-step in the trading period, the position values after close are used to calculate returns on particular assets and P&L for $t$.

### Hyper-Optimization

The strategy requires specification of factors such as: (a) baskets to perform PCA on, (b) assets to be traded, (c) number of PCs to construct systematic components, (d) orders of lags in ARMA models. In order to select them, a hyper-optimization is performed on the in-sample rolling window during Walk-Forward Analysis (WFA). The general procedure of hyper-optimization is the following:
1. A Genetic Optimization Algorithm is run to search the baskets space for a plateau of high performance in a trading simulation with simplified position sizing and return calculation.
2. For the selected basket, ARMA residual predictions are made for all combinations of lag orders and numbers of PCs used to construct systematic components.
3. A more detailed trading simulation is performed to search the parameter space for plateaus of high returns on individually traded assets.
4. Candidates selected in the previous step are used to find the best performing combination of assets and parameters in a fully detailed trading simulation.

#### Basket Selection

The space of baskets of assets used for PCA transformation is searched via Genetic Optimization Algorithm. Baskets are represented by individuals (solutions) which compose populations in subsequent generations. The algorithm is meant to ensure that each new generation involves individuals that are similar to "the fittest" ones (in our case - the ones providing the best performance) from the preceding generation, and simultaneously, the space is well explored. A basket is a solution encoded by individual's genes, that is, a binary sequence (encoding) of elements corresponding to each asset which can be included or excluded from the PCA transformation (a set of assets which are always included in the basket, i.e., 'fixed assets', can be specified). The initial population is generated by randomly drawing 0 or 1 for each element, such that the number of assets in the basket is within a specified range. Each individual's fitness is evaluated based on performance of the strategy for a given basket and neighbouring baskets. Fitter individuals have a higher probability of propagation. Majority of the new generation is created by applying crossover and mutation on selected parents' genes. The rest is filled with randomly generated immigrants. The population size is constant over generations. The process is repeated for a specified number of iterations. The algorithm has been designed specifically to backtest the presented strategy, but was partially inspired by the approach of Brabazon and O'Neill (2006).

Specific steps taken in each iteration are as follows. For each individual, two neighbours are picked - one by a single swap of two random elements (an exchange of 0 and 1 on their positions), and one by a single switch of a random element (changing 0/1 to the opposite). Then, a simplified trading simulation is run for the basket encoded by given individual's genes (main basket), as well as for the baskets encoded by it's neighbours (neighbour baskets). The simplified simulation: 
- uses AR(2) model (estimated using Yule-Walker Equations) for residual predictions,
- does not take correlations into account (as if assets were individually traded),
- assumes:
  - constant portfolio value when calculating price impact, 
  - realized costs are equal to predicted costs,
  - positions are always fully closed after one time-step.

Such simulation is performed for each considered number of PCs used for systematic component construction. Once the portfolio returns for each simulation are obtained, a few top returns earned on each asset are removed (replaced with 0), and a cumulative return is obtained for each asset. Next, these cumulative returns are smoothed in the PCs number dimension by taking a weighted average of cumulative returns for a given number and 2 closest numbers. The number of PCs with the highest smoothed cumulative return, along with it's neighbours (2 closest numbers), is selected for each asset. For each asset, the returns time series corresponding to the selected number of PCs is taken, and they are combined together. Analogous procedure is performed for returns corresponding to the neighbouring numbers. Only the assets required to be included (fixed assets), or those whose top smoothed cumulative return exceeds certain threshold, are considered in this step. Sharpe ratio with concentration penalty is obtained for each of these sets of returns (summed for each time step). The concentration penalty is calculated as the biggest drop in Sharpe ratio due to removal of a single asset multiplied by some parameter. A weighted average of these modified Sharpe ratios is the performance metric corresponding to a given basket. A weighted average of performance metrics corresponding to the main basket and it's neighbour baskets is the fitness of a given individual.

Individuals are ranked based on their fitness. One or two best individuals are passed to the new (next generation's) population. Certain fraction of the new population will be filled with randomly generated immigrants. The rest of the new population is formed by children created based on parents' genes. For this purpose, a respective number of parents pairs is selected. Each parent in each pair can be selected in two ways: by drawing a single individual according to selection probabilities, or via tournament selection. Selection probabilities are assigned based on a negative exponential function of individuals' ranks. Tournament selection is conducted by uniformly drawing a certain number of individuals and taking the fittest one. Two children are produced from each pair of parents by copying parents' genes and applying crossover and mutation on them. Crossover is performed by taking elements that differ for both children, and swapping them with some probability. Mutation is performed by randomly picking an element of a given child's genes and changing it to the opposite (switch) with some probability, or by picking two different elements and exchanging them (swap) with some probability.

Once the algorithm goes through all iterations, top individuals from the last generation's population are taken, and the neighbourhoods of the corresponding baskets are explored in depth. For each individual, the performance is assessed for a much larger number of neighbour baskets (e.g., 100). A weighted average of performance metrics within each neighbourhood is obtained, along with other statistics. These results will be used in WFA to select the final basket, passed to further steps of hyper-optimization.

#### Traded Assets, PCs Numbers and ARMA Orders Selection

For the selected basket, predictions of residuals in the in-sample period are generated for all assets which can be potentially traded, all considered numbers of PCs used to construct systematic components, and all considered orders of lags in the ARMA model.

For each generated series of residual predictions, a trading simulation is run, where assets are individually traded. The simulation is the same as described in [Trading and Return Calculation](/README.md#trading-and-return-calculation), except each asset has it's own "account". That is, when the positions are taken for particular assets, correlations between them are not taken into account. Each asset has it's own (scalar-only) formula for optimal weight $w_i^*$, where the covariance matrix is replaced with respective variance. Value of the portfolio corresponding to a given asset is only impacted by positions taken on this asset. Similarly as in case of basket selection, few highest returns earned on each asset are removed. Next, cumulative returns are obtained and smoothed along the PCs number and ARMA lags orders dimensions. For this purpose, closeness of lags orders must be defined. For each pair of orders, the other pairs are classified into 3 categories according to their similarity. Smoothing is applied by taking a weighted average of a given cumulative return and returns for different orders and the same or similar number of PCs. Returns for similar orders and the same PCs number, or vice versa, have the highest weights (after the main return), and the weights decrease as proximity decreases. Based on these results, candidate PCs number & orders combinations for particular assets (as well as assets themselves) will be selected in WFA to further test their performance in joined trading.

Candidate PCs number & orders combinations for particular assets are explored through the prioritized search algorithm. The algorithm goes through each candidate combination for one asset, while combinations for other assets remain unchanged. Combination yielding the highest value of the objective function is now set for this asset, and the algorithm moves to the next one. The procedure is repeated for all assets. Then, the algorithm starts a new round, going through each asset once again. It only stops if no new best combination of combinations has been found in a given round. The objective function is a weighted average of performance metrics within the neighbourhood of a given combination of combinations. It is obtained in the following way. A full trading simulation (as described in [Trading and Return Calculation](/README.md#trading-and-return-calculation)) is run for the current combination of combinations, and few top returns are removed for each asset, similarly as before. The corresponding performance metric is calculated, using total returns (returns summed for each time step) obtained in this simulation. The metric is defined as a Sharpe ratio with concentration penalty, calculated the same way as in case of basket selection. Next, a specified number neighbours of the current combination of combinations is selected randomly for each of 3 categories, corresponding to different levels of proximity in the PCs number and orders space, defined beforehand (analogous as in the previous step). For each neighbour, the trading simulation is run, and performance metric is obtained in the same way as for the main combination of combinations. The value of the objective function is calculated by taking the weighted average of performance metric of the main combination of combinations and mean performance metrics in each neighbourhood category. This value is saved for each considered combination of combinations, along with other statistics of performance metrics. Based on them, the final combination of combinations will be selected in WFA and used in the out-of-sample period.

### Walk-Forward Analysis

Walk-Forward Analysis (WFA) is organized such that hyper-optimization is conducted on the in-sample rolling window to fine-tune hyper-parameters, and trading simulation is performed on the out-of-sample rolling window for the previously selected hyper-parameters. In-sample and out-of-sample windows must be distinguished from calibration and deployment windows. The purpose of the rolling calibration window is to construct PCA projection matrix, estimate ARMA models, and estimate correlation matrix, which are then used in a trading simulation on the rolling deployment window. In-sample and out-of-sample rolling windows are composed of subsequent deployment periods. Baskets for PCA transformation, ARMA lag orders, numbers of PCs to construct systematic components, and assets to be traded in the subsequent out-of-sample periods are selected based on the performance they yield over the preceding in-sample periods. Assume calibration window has length $\Delta t_{cal}$, deployment window has length $\Delta t_{dep}$, in-sample window is composed of $N$ deployment windows, and out-of-sample window is composed of $n$ deployment periods. Consider a single walk-forward (1 in-sample period + 1 out-of-sample period), whose out-of-sample period starts at $t_0$. The corresponding in-sample period will cover deployment periods from $t_0 - N \times \Delta t_{dep}$ to $t_0$, and the in-sample period wil cover deployment periods from $t_0$ to $t_0 + n \times \Delta t_{dep}$. A diagram illustrating order of particular windows for such a walk-forward is presented below. Note that the first deployment period uses estimations from a calibration period outside of the in-sample period. Thus, for an out-of-sample period to begin at $t_0$, data from the $\left[ t_0 - \left(\Delta t_{IS} + \Delta t_{cal} \right) \right]$ interval must be utilized to conduct full hyper-optimization.

<img width="1216" height="765" alt="image" src="https://github.com/user-attachments/assets/705d5bd9-3593-4fd9-9421-74b1bb94eded" />

Such walk-forwards are rolled for the entire backtesting period. That is, subsequent out-of-sample periods start where the preceding ones ended, and the in-sample window is shifted accordingly, until the entire backtesting period is covered. The last out-of-sample period may be shorter, if the previous one ended less than $\Delta t_{OoS}$ before the end of backtesting period. 

The backtesting presented in this repository covers period from 22.01.2025 to 30.09.2025. The following window lengths have been set for the WFA run in the XXX notebook:
- calibration window ($\Delta t_{cal}$): 30 days,
- deployment window ($\Delta t_{dep}$): 3 days,
- in-sample window ($\Delta t_{IS}$): 90 days,
- out-of-sample window ($\Delta t_{OoS}$): 30 days.

The specific steps taken in each walk-forward are as follows. First, data for respective periods is filtered. Then, Genetic Algorithm is run on the in-sample period, and extended performance statistics for top individuals from the last generation are obtained, as described in [Basket Selection](/README.md#basket-selection). The final basket is selected based on the weighted average of performance metrics in the neighbourhood. However, in order to avoid spikes in the searched space, additional conditions/corrections may be imposed. This includes:
- (iteratively) filtering out baskets for which the performance metric exceeds the average of performance metrics in it's neighbourhood, unless:
  - too many of them are removed, or
  - all of the top ones are removed;
- adding a penalty proportional to the difference between performance of the main basket and average performance in it's neighbourhood.

The latter option has been applied in the presented backtesting.

Once the basket is selected, ARMA predictions are generated for all considered numbers of PCs for systematic component construction and lags orders, and cumulative returns earned in individual trading are obtained for each asset and combination, as described in [Traded Assets, PCs Numbers and ARMA Orders Selection](/README.md#traded-assets-pcs-numbers-and-arma-orders-selection).

### Post-Analysis

e

### Further Steps

h

### Bibliography

Almgren, R., Thum, C., Hauptmann, E., Li, H. (2005). <i>Direct Estimation of Equity Market Impact.</i> Risk, 18(7), 58-62.

Brabazon, A., O'Neill, M. (2006). <i>Biologically Inspired Algorithms for Financial Modelling.</i> Berlin and Heidelberg: Springer. 

Huang, R. D., Stoll, H. R. (1997). <i>The Components of the Bid-Ask Spread: A General Approach.</i> The Review of Financial Studies, 10(4), 995-1034.

Roll, R. (1984). <i>A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market.</i> The Journal of Finance, 39(4), 1127-1139.

Stoll, H. R. (1989). <i>Inferring the Components of the Bid-Ask Spread: Theory and Empirical Tests.</i> The Journal of Finance, 44, 115-134.
