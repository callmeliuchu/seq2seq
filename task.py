#!/usr/bin/env python
# coding: utf-8

# # test1

# ## 输入数据
# 
# | 入库日期      | 产品 | 数量 |
# |--------------|------|------|
# | 2024/9/30    | A    | 10   |
# | 2024/10/1    | A    | 1    |
# | 2024/10/22   | A    | 5    |
# | 2024/11/1    | A    | 2    |
# | 2024/11/25   | A    | 7    |
# | 2024/12/1    | A    | 3    |
# | 2025/1/1     | A    | 4    |
# | 2024/9/30    | B    | 100  |
# | 2024/10/1    | B    | 10   |
# | 2024/11/1    | B    | 20   |
# | 2024/12/1    | B    | 30   |
# | 2025/2/2     | B    | 40   |

# In[1]:


import pandas as pd

# Creating the structured DataFrame based on the provided data
data = {
    "入库日期": [
        "2024/9/30", "2024/10/1", "2024/10/22", "2024/11/1", "2024/11/25", "2024/12/1", "2025/1/1",
        "2024/9/30", "2024/10/1", "2024/11/1", "2024/12/1", "2025/2/2"
    ],
    "产品": ["A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    "数量": [10, 1, 5, 2, 7, 3, 4, 100, 10, 20, 30, 40]
}


# In[2]:


from datetime import datetime


def format_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y/%m/%d')
    year_month = f"{date_obj.year}_{date_obj.month:02d}"
    return year_month


# In[3]:


products = list(set(data['产品']))


# In[4]:


products


# In[5]:


year_month_count = {}
for date,product,count in zip(data['入库日期'],data['产品'],data['数量']):
    print(date,product,count)
    year,month,day = date.split('/')
#     print(year,month,day)
    key = format_date(date)
    if key not in year_month_count:
        year_month_count[key] = {}
    year_month_count[key][product] = year_month_count[key].get(product,0) + count


# In[6]:


year_month_count


# In[7]:


keys = list(year_month_count.keys())


# In[8]:


keys


# In[9]:


keys.sort()


# In[10]:


keys


# In[11]:


for key in keys:
    for pro in products:
        if pro not in year_month_count[key]:
            year_month_count[key][pro] = 0
#     year_month_count[key]['total'] = sum(v for k,v in year_month_count[key].items())


# In[12]:


year_month_count


# In[13]:


stages_count = {}


# In[14]:


for key in keys:
    init = {pro:0 for pro in products}
    final = {pro:0 for pro in products}
    stages_count[key] = {'current': year_month_count[key],'init':init,'final':final}


# In[15]:


stages_count


# In[ ]:





# In[16]:


for i,key in enumerate(keys):
    ikeys = stages_count[key]['final'].keys()
    if i > 0:
        for ik in ikeys:
            stages_count[key]['init'][ik] = stages_count[keys[i-1]]['final'][ik]
    for ik in ikeys:
        stages_count[key]['final'][ik] = stages_count[key]['init'][ik] + stages_count[key]['current'][ik]
        


# In[17]:


stages_count


# In[18]:


for key in keys:
    for stage in stages_count[key]:
        stages_count[key][stage]['total'] = sum(v for k,v in stages_count[key][stage].items())


# In[19]:


stages_count


# In[20]:


flatten_data = {}


# In[21]:


for key in keys:
    for stage in stages_count[key]:
        if stage == 'init':
            new_key = key + '期初库存'
        elif stage == 'current':
            new_key = key + '本月数量'
        else:
            new_key = key + '期末库存'
        flatten_data[new_key] = stages_count[key][stage]


# In[22]:


flatten_data


# In[23]:


import pandas as pd


# In[24]:


pd.DataFrame(flatten_data)


# # test2

# In[25]:


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
2023-12	67771	1851000"""


# In[26]:


data = {}
data['date'] = []
data['count'] = []
for line in text.split('\n')[1:]:
    ym,code,count = line.split('\t')
#     print(ym,code,count)
    data['date'].append(ym)
    data['count'].append(int(count))


# In[27]:


df = pd.DataFrame(data)


# In[28]:


df


# # 方法1

# ARIMA（AutoRegressive Integrated Moving Average，自回归积分移动平均模型）是一种流行的时间序列模型，常用于预测平稳时间序列数据。ARIMA 模型通过自回归（AR）、差分（I）和移动平均（MA）三部分组合来捕捉时间序列的模式。
# 
# ### ARIMA 组成部分
# 
# ARIMA 模型由三个主要参数组成，分别表示模型中的阶数：
# 1. **AR（自回归）**：`p` 表示自回归项的阶数。它表示预测值与前几期数据之间的关系。例如，当 `p=1` 时，模型会使用前一期的数据来预测当前值。
# 2. **I（差分）**：`d` 表示差分次数。差分是对非平稳数据进行平稳化处理的技术。`d` 值表示差分操作的次数，例如 `d=1` 表示对数据进行一次差分。
# 3. **MA（移动平均）**：`q` 表示移动平均项的阶数。移动平均项基于前几期的预测误差来调整当前预测值。例如，当 `q=1` 时，模型使用前一期的误差进行预测修正。
# 
# ### ARIMA 参数选择
# 
# - **(p, d, q) 参数**：ARIMA 模型的关键在于选择合适的参数 `(p, d, q)`。这些参数通常通过网格搜索或 AIC/BIC 值最小化来确定。
# - **AIC（赤池信息准则）** 和 **BIC（贝叶斯信息准则）**：用于模型的优劣评价，较低的 AIC/BIC 值表示更优的模型。
# 
# ### ARIMA 的模型表示
# 
# ARIMA 模型预测方程通常表示为：
# 
# \[
# Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + \dots + \phi_p Y_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \dots + \theta_q \epsilon_{t-q} + \epsilon_t
# \]
# 
# 其中：
# - \( Y_t \) 是时间 \( t \) 的预测值。
# - \( \phi \) 和 \( \theta \) 是 AR 和 MA 部分的系数。
# - \( \epsilon_t \) 是噪声项。
# 
# ### ARIMA 应用步骤
# 
# 1. **数据准备**：通常需要先将时间序列数据转换为平稳形式（均值和方差不随时间变化），可以通过差分来实现。
# 2. **模型训练**：使用选定的 `(p, d, q)` 参数拟合 ARIMA 模型。
# 3. **模型评估**：根据 AIC/BIC 值和残差分析评估模型的效果。
# 4. **生成预测**：使用拟合的 ARIMA 模型对未来数据进行预测。
# 
# ### ARIMA 优缺点
# 
# **优点：**
# - ARIMA 模型适用于较短期的时间序列预测，能有效捕捉线性趋势和周期性波动。
# - 模型解释性强，适合分析时间序列的结构。
# 
# **缺点：**
# - ARIMA 模型只能处理单变量时间序列。
# - 对非线性数据效果较差，且参数选择较为复杂，需要对数据进行大量处理和调优。
# 
# ### 总结
# 
# ARIMA 是时间序列分析的经典模型，适合平稳、单变量时间序列的数据预测需求。对于数据中存在的趋势和季节性，可以考虑 ARIMA 的扩展模型 SARIMA（季节性 ARIMA）。

# In[29]:


from statsmodels.tsa.arima.model import ARIMA


# In[30]:


# 定义 ARIMA 模型
model = ARIMA(df['count'], order=(1, 1, 1))
# 拟合模型
fitted_model = model.fit()


# In[31]:


fitted_model.summary()


# In[32]:


forecast = fitted_model.get_forecast(steps=3)


# In[33]:


forecast_mean = forecast.predicted_mean


# In[ ]:





# In[34]:


preds = [int(d) for d in forecast_mean.tolist()]


# '2024-01','2024-02','2024-03' 预测

# In[35]:


preds


# In[36]:


dates = ['2024-01','2024-02','2024-03']


# In[37]:


dates = list(df['date'].values) + dates


# In[38]:


values = list(df['count'].values) + preds


# In[39]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(dates, values, marker='o', linestyle='-', color='b', label='Sales')
plt.title("Monthly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # 旋转日期标签
plt.tight_layout()  # 调整布局以适应旋转的标签
plt.show()


# # 方法2

# `Prophet` 是一个由 Facebook（现 Meta）开发的开源时间序列预测工具。它特别适合用于包含趋势和季节性变化的时间序列数据，具有处理缺失数据和异常值的鲁棒性。由于其简单的接口和良好的预测效果，`Prophet` 被广泛应用于业务领域，如需求预测、流量预测等。
# 
# ### Prophet 的优势
# 
# 1. **简单易用**：Prophet 对用户非常友好，提供的接口使得用户只需少量的代码就能得到高质量的预测结果。
# 2. **适应性强**：它能够很好地处理包含节假日、周末、季节性变化等因素的数据，同时支持多种时间粒度的数据（如天、周、月等）。
# 3. **鲁棒性**：Prophet 对缺失值、不规则时间间隔的数据以及异常值具有很强的鲁棒性，能够适应较为复杂的业务需求。
# 
# ### Prophet 的核心概念
# 
# Prophet 模型基于以下公式来分解时间序列：
# \[
# y(t) = g(t) + s(t) + h(t) + \epsilon_t
# \]
# 其中：
# - \( g(t) \) 表示趋势，描述数据的长期增长或减少趋势。
# - \( s(t) \) 表示季节性变化，描述周期性波动（如周、年季节性）。
# - \( h(t) \) 表示节假日效应，表示特殊日期的影响。
# - \( \epsilon_t \) 是误差项。
# 
# ### Prophet 的主要功能
# 
# - **趋势预测**：Prophet 可以自动识别数据的增长或衰退趋势，并支持线性和非线性趋势。
# - **季节性变化**：支持按年、按周、按日的季节性变化。
# - **节假日建模**：允许用户自定义节假日和特殊事件，以考虑其对数据的影响。
# - **不确定性范围**：输出预测值的置信区间，帮助用户更好地评估预测的准确性。

# In[40]:


from prophet import Prophet
# df_prophet = df.reset_index().rename(columns={'index': 'ds', 'sales': 'y'})
# model = Prophet()
# model.fit(df_prophet)
# future = model.make_future_dataframe(periods=12, freq='M')
# forecast = model.predict(future)


# In[41]:


data = df.rename(columns={'date': 'ds', 'count': 'y'})


# In[42]:


data


# In[43]:


model = Prophet()

# Fit the model
model.fit(data)

# Make future DataFrame for predictions (e.g., forecast for 12 months ahead)
future = model.make_future_dataframe(periods=3, freq='M')

# Predict future values
forecast = model.predict(future)

# Display forecasted values
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])


# In[44]:


preds = [int(d) for d in forecast[-3:]['trend'].values]


# '2024-01','2024-02','2024-03' 预测

# In[45]:


preds


# In[46]:


values = list(df['count'].values) + preds


# In[47]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 创建折线图
plt.figure(figsize=(10, 6))
plt.plot(dates, values, marker='o', linestyle='-', color='b', label='Sales')
plt.title("Monthly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # 旋转日期标签
plt.tight_layout()  # 调整布局以适应旋转的标签
plt.show()


# In[ ]:




