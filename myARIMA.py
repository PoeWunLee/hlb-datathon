from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import relativedelta


def thisARIMA(myDF, iterDict, rolling_average, num_forecast):
    '''
    Returns time series forecasting value
        Parameters:
            myDF (data frame): cleaned master dataframe
            iterDict (dictionary): dictionary of user's input
                                    for eg:  iterDict = {"Residential_Type": restype,
                                                         "Property_State": state,
                                                         "Property_Type": proptype,
                                                          "Landed_Type": land}
            num_forecast (int): number of forecast
            rolling_average (list): rolling average of the average of difference
        Returns:
            mth (list) = months (x values of the plot)
            output_y (list) = rolling average + forecast value (y value of the plot)
    }
    '''

    for key, value in iterDict.items():
        myDF = myDF.loc[myDF[key] == value]

    sumdf = myDF.groupby(["Submission_Mth"]).mean().reset_index()

    model = SARIMAX(sumdf.difference, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    results = model.fit(disp=-1)
    forecast = results.forecast(steps=num_forecast)

    mth = [datetime.strptime(date, '%Y-%m') for date in sumdf.Submission_Mth.tolist()]

    for _ in range(num_forecast):
        mth.append(mth[-1] + relativedelta(months=1))

    output_y = rolling_average + forecast.tolist()

    return mth, output_y

