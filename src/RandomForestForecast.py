import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

sp500 = yf.Ticker('^GSPC')
sp500 = sp500.history(period='max')

print(sp500.tail())

del sp500["Dividends"]
del sp500["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1)
# 1 if the price goes up, 0 if it goes down
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
print(sp500.tail())

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)
def createPredictors(data):
    horizons = [2,5,60,250,1000]
    predictors = []

    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        
        predictors+= [ratio_column, trend_column]
    return predictors
predictors = createPredictors(sp500)
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
print(sp500.tail())

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

predictions = backtest(sp500, model, predictors)
print(precision_score(predictions["Target"], predictions["Predictions"]))
print(predictions.tail())
