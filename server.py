import pandas as pd
from datetime import datetime
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt

# 设置页面标题
st.title("数据分析与预测展示")

# 销售数据
sales_data_text = """销售年月	产品代码	销售数量
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
2023-12	67771	1851000"""

# 将文本数据转换为 DataFrame
sales_data = []
for line in sales_data_text.split('\n')[1:]:
    ym, code, count = line.split('\t')
    sales_data.append([ym, int(code), int(count)])
df_sales = pd.DataFrame(sales_data, columns=["date", "product_code", "count"])
df_sales['date'] = pd.to_datetime(df_sales['date'], format='%Y-%m')
df_sales.set_index('date', inplace=True)

# 库存数据
inventory_data = {
    "入库日期": [
        "2024/9/30", "2024/10/1", "2024/10/22", "2024/11/1", "2024/11/25", "2024/12/1", "2025/1/1",
        "2024/9/30", "2024/10/1", "2024/11/1", "2024/12/1", "2025/2/2"
    ],
    "产品": ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    "数量": [10, 1, 5, 2, 7, 3, 4, 100, 10, 20, 30, 40]
}


# 定义 ARIMA 和 Prophet 模型的预测函数
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


# 定义库存数据处理函数
def process_inventory_data(data):
    def format_date(date_str):
        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
        return f"{date_obj.year}_{date_obj.month:02d}"

    products = list(set(data['产品']))
    year_month_count = {}
    for date, product, count in zip(data['入库日期'], data['产品'], data['数量']):
        key = format_date(date)
        if key not in year_month_count:
            year_month_count[key] = {}
        year_month_count[key][product] = year_month_count[key].get(product, 0) + count

    keys = sorted(year_month_count.keys())
    for key in keys:
        for pro in products:
            if pro not in year_month_count[key]:
                year_month_count[key][pro] = 0

    stages_count = {}
    for key in keys:
        init = {pro: 0 for pro in products}
        final = {pro: 0 for pro in products}
        stages_count[key] = {'current': year_month_count[key], 'init': init, 'final': final}

    for i, key in enumerate(keys):
        if i > 0:
            for pro in products:
                stages_count[key]['init'][pro] = stages_count[keys[i - 1]]['final'][pro]
        for pro in products:
            stages_count[key]['final'][pro] = stages_count[key]['init'][pro] + stages_count[key]['current'][pro]

    for key in keys:
        for stage in stages_count[key]:
            stages_count[key][stage]['total'] = sum(stages_count[key][stage].values())

    flatten_data = {}
    for key in keys:
        for stage, values in stages_count[key].items():
            stage_name = f"{key}期初库存" if stage == 'init' else f"{key}期末库存" if stage == 'final' else f"{key}本月数量"
            flatten_data[stage_name] = values
    return flatten_data


# 页面内容选择
task = st.radio("选择任务", ["销售数据预测", "库存数据分析"])

if task == "销售数据预测":
    st.subheader("销售数据预测")

    # 选择模型
    model_choice = st.radio("选择预测模型", ["ARIMA", "Prophet"])

    # 根据选择的模型进行预测
    if model_choice == "ARIMA":
        preds = forecast_arima(df_sales)
    elif model_choice == "Prophet":
        preds = forecast_prophet(df_sales)

    # 显示预测结果
    forecast_dates = pd.date_range(start=df_sales.index[-1], periods=4, freq='M')[1:]
    all_dates = df_sales.index.strftime('%Y-%m').tolist() + [date.strftime('%Y-%m') for date in forecast_dates]
    all_values = list(df_sales['count'].values) + preds

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(all_dates, all_values, marker='o', linestyle='-', label='Sales')
    plt.title("Monthly Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

    # 显示预测数据表格
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Sales': preds})
    st.write("预测的销量：")
    st.write(forecast_df)

elif task == "库存数据分析":
    st.subheader("Inventory Data Analysis")

    # Process the inventory data
    flatten_data = process_inventory_data(inventory_data)

    # Convert processed data to DataFrame and transpose for easier plotting
    flatten_df = pd.DataFrame(flatten_data).T
    st.write("Inventory Analysis Results")
    st.write(flatten_df)

    # Extract unique month labels for plotting
    month_labels = sorted(set(key.split('期')[0] for key in flatten_df.index if '期' in key))

    # Initialize lists for storing values for each inventory stage
    a_init_values, b_init_values, total_init_values = [], [], []
    a_current_values, b_current_values, total_current_values = [], [], []
    a_final_values, b_final_values, total_final_values = [], [], []

    # Loop through each month and extract values for each stage (期初库存, 本月数量, 期末库存)
    for month in month_labels:
        # Extract beginning inventory (期初库存)
        init_row = flatten_df.loc[flatten_df.index.str.startswith(month) & flatten_df.index.str.contains('期初库存')]
        if not init_row.empty:
            a_init_values.append(init_row['A'].values[0])
            b_init_values.append(init_row['B'].values[0])
            total_init_values.append(init_row['total'].values[0])
        else:
            a_init_values.append(0)
            b_init_values.append(0)
            total_init_values.append(0)

        # Extract current month inventory (本月数量)
        current_row = flatten_df.loc[flatten_df.index.str.startswith(month) & flatten_df.index.str.contains('本月数量')]
        if not current_row.empty:
            a_current_values.append(current_row['A'].values[0])
            b_current_values.append(current_row['B'].values[0])
            total_current_values.append(current_row['total'].values[0])
        else:
            a_current_values.append(0)
            b_current_values.append(0)
            total_current_values.append(0)

        # Extract ending inventory (期末库存)
        final_row = flatten_df.loc[flatten_df.index.str.startswith(month) & flatten_df.index.str.contains('期末库存')]
        if not final_row.empty:
            a_final_values.append(final_row['A'].values[0])
            b_final_values.append(final_row['B'].values[0])
            total_final_values.append(final_row['total'].values[0])
        else:
            a_final_values.append(0)
            b_final_values.append(0)
            total_final_values.append(0)

    # Ensure data consistency before plotting
    if len(month_labels) == len(a_init_values) == len(b_init_values) == len(total_init_values):
        # Plot Beginning Inventory (期初库存)
        plt.figure(figsize=(14, 6))
        x = range(len(month_labels))
        bar_width = 0.25
        plt.bar([p - bar_width for p in x], a_init_values, width=bar_width, color='blue', label='A')
        plt.bar(x, b_init_values, width=bar_width, color='green', label='B')
        plt.bar([p + bar_width for p in x], total_init_values, width=bar_width, color='red', label='Total')
        plt.title("Monthly Beginning Inventory Levels")
        plt.xlabel("Month")
        plt.ylabel("Quantity")
        plt.xticks(x, month_labels, rotation=45)
        plt.legend()
        st.pyplot(plt)

    if len(month_labels) == len(a_current_values) == len(b_current_values) == len(total_current_values):
        # Plot Current Month Inventory (本月数量)
        plt.figure(figsize=(14, 6))
        plt.bar([p - bar_width for p in x], a_current_values, width=bar_width, color='blue', label='A')
        plt.bar(x, b_current_values, width=bar_width, color='green', label='B')
        plt.bar([p + bar_width for p in x], total_current_values, width=bar_width, color='red', label='Total')
        plt.title("Monthly Inventory Inflow")
        plt.xlabel("Month")
        plt.ylabel("Quantity")
        plt.xticks(x, month_labels, rotation=45)
        plt.legend()
        st.pyplot(plt)

    if len(month_labels) == len(a_final_values) == len(b_final_values) == len(total_final_values):
        # Plot Ending Inventory (期末库存)
        plt.figure(figsize=(14, 6))
        plt.bar([p - bar_width for p in x], a_final_values, width=bar_width, color='blue', label='A')
        plt.bar(x, b_final_values, width=bar_width, color='green', label='B')
        plt.bar([p + bar_width for p in x], total_final_values, width=bar_width, color='red', label='Total')
        plt.title("Monthly Ending Inventory Levels")
        plt.xlabel("Month")
        plt.ylabel("Quantity")
        plt.xticks(x, month_labels, rotation=45)
        plt.legend()
        st.pyplot(plt)
    else:
        st.error("Data length mismatch: Unable to plot all charts.")

