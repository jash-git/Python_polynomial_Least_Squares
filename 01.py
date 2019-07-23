#--------------------------#
#Step 1：导入相关库和类
import pandas as pd
import tushare as ts
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#--------------------------#

#--------------------------#
#Step 2：读取数据
# 获取沪深300，中金细分金融指数2018年全年日线数据及2019年前7个交易日数据
index_hs300 = pro.index_daily(ts_code='399300.SZ', start_date='20180101', end_date='20190110')
index_finance = ts.pro_bar(ts_code='000818.CSI', freq='D', asset='I', start_date='20180101', end_date='20190110')

index_hs300_pct_chg = index_hs300.set_index('trade_date')['pct_chg']
index_finance_pct_chg = index_finance.set_index('trade_date')['pct_chg']
df = pd.concat([index_hs300_pct_chg, index_finance_pct_chg], keys=['y_hs300', 'x_finance'], join='inner', axis=1, sort=True)

df_existing_data = df[df.index < '20190101']

x = df_existing_data[['x_finance']]
y = df_existing_data['y_hs300']
#--------------------------#

#--------------------------#
#Step 3：自变量转换
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
#--------------------------#

#--------------------------#
#Step 4：建立模型并拟合数据
model = LinearRegression().fit(x_, y)
#--------------------------#

#--------------------------#
#Step 5：输出结果
r_sq = model.score(x_, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
#--------------------------#

