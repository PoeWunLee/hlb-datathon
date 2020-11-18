from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import relativedelta


def thisARIMA(past_values, mth, rolling_average, num_forecast):
    '''
    Returns time series forecasting value
        Parameters:
            past_values (list): previous average values
            mth (list) : previous months
            num_forecast (int): number of forecast
            rolling_average (list): rolling average of the average of difference
        Returns:
            mth (list) = months (x values of the plot)
            output_y (list) = rolling average + forecast value (y value of the plot)
    }
    '''
    model = SARIMAX(past_values, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=-1)
    forecast = results.forecast(steps=num_forecast)

    mth = [datetime.strptime(date, '%Y-%m') for date in mth]

    for _ in range(num_forecast):
        mth.append(mth[-1] + relativedelta(months=1))

    output_y = forecast.tolist()
    mth = mth[-len(output_y):]

    return mth, output_y

