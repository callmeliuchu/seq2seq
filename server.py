import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from io import StringIO

# 设置页面标题
st.title("销售预测模型对比")

# 输入数据区域
st.subheader("输入销售数据")
text = """销售年月	产品代码	销售数量
2019-01	67771	942500
2019-02	67771	685000
2019-03	67771	1247500
2019-04	67771	940000
2019-05	67771	1287500
2019-06	67771	1527500
2019-07	67771	1072500
2019-08	67771	1472500
2019-09	67771	1252500
2019-10	67771	1472500
2019-11	67771	1075000
2019-12	67771	1320207
2020-01	67771	1589793
2020-02	67771	888411
2020-03	67771	1461119
2020-04	67771	2491448
2020-05	67771	2506326
2020-06	67771	1335857
2020-07	67771	2042500
2020-08	67771	1487500
2020-09	67771	1841839
2020-10	67771	2080000
2020-11	67771	2615000
2020-12	67771	2852500
2021-01	67771	3840000
2021-02	67771	1260901
2021-03	67771	1977500
2021-04	67771	1318354
2021-05	67771	1811989
2021-06	67771	1764412
2021-07	67771	1853745
2021-08	67771	2416519
2021-09	67771	2968162
2021-10	67771	2022431
2021-11	67771	3226145
2021-12	67771	4080000
2022-01	67771	1895000
2022-02	67771	3217500
2022-03	67771	1687277
2022-04	67771	2847500
2022-05	67771	1777500
2022-06	67771	3735000
2022-07	67771	677500
2022-08	67771	710000
2022-09	67771	1810000
2022-10	67771	15150
2022-12	67771	5000
2023-02	67771	0
2023-03	67771	517500
2023-04	67771	747500
2023-05	67771	1072500
2023-06	67771	1607500
2023-07	67771	1450500
2023-08	67771	1837500
2023-09	67771	1696000
2023-10	67771	1998000
2023-11	67771	1203000
2023-12	67771	1851000"""  # 省略部分数据，需提供完整数据
data_text = st.text_area("请输入销售数据", text)

# 解析输入数据
data = {'date': [], 'count': []}
for line in data_text.split('\n')[1:]:
    ym, code, count = line.split('\t')
    data['date'].append(ym)
    data['count'].append(int(count))
df = pd.DataFrame(data)

# 数据准备
df['date'] = pd.to_datetime(df['date'], format='%Y-%m')
df.set_index('date', inplace=True)

# 选择模型
st.subheader("选择预测模型")
model_choice = st.radio("选择模型", ["ARIMA", "Prophet"])

# 定义预测函数
def forecast_arima(data, steps=3):
    model = ARIMA(data['count'], order=(1, 1, 1))
    fitted_model = model.fit()
    forecast = fitted_model.get_forecast(steps=steps)
    return forecast.predicted_mean.tolist()

def forecast_prophet(data, steps=3):
    prophet_data = data.reset_index().rename(columns={'date': 'ds', 'count': 'y'})
    model = Prophet()
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=steps, freq='M')
    forecast = model.predict(future)
    return forecast['yhat'][-steps:].tolist()

# 根据选择的模型进行预测
if model_choice == "ARIMA":
    preds = forecast_arima(df)
elif model_choice == "Prophet":
    preds = forecast_prophet(df)

# 展示预测结果
dates = pd.date_range(start=df.index[-1], periods=4, freq='M')[1:]
predicted_dates = [date.strftime('%Y-%m') for date in dates]
forecasted_values = list(df['count'].values) + preds

# 绘图
st.subheader("预测结果")
plt.figure(figsize=(10, 6))
plt.plot(df.index.strftime('%Y-%m').tolist() + predicted_dates, forecasted_values, marker='o', linestyle='-', label='Sales')
plt.title("Monthly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

# 显示预测结果表格
st.write("预测的销量：")
forecast_df = pd.DataFrame({'日期': predicted_dates, '预测销量': preds})
st.table(forecast_df)
