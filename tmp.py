from neuralprophet import NeuralProphet
# Input Data

import pandas as pd
df = pd.read_csv(
    'C:/Users/DataKnow/Documents/neural_prophet/example_data/wp_log_peyton_manning.csv'
    )
df.head()
# Simple Model
m = NeuralProphet(daily_seasonality=True)
metrics = m.fit(df, freq="D")
# Future
future = m.make_future_dataframe(df, periods=365)
forecast = m.predict(future)
forecasts_plot = m.plot(forecast)
# Por componentes
fig_comp = m.plot_components(forecast)
# Coeficientes individuales
fig_param = m.plot_parameters()
# Validacion del modelo 
# Se puede hacer por dos frentes: Split de datos, por cada eproch
m = NeuralProphet(daily_seasonality=True)
# Frente 1
df_train, df_val = m.split_df(df, valid_p=0.2)
train_metrics = m.fit(df_train, freq="D")
val_metrics = m.test(df_val)
# Frente 2
m = NeuralProphet(daily_seasonality=True)
metrics = m.fit(df, freq="D", validate_each_epoch=True, valid_p=0.2)

