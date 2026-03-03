import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)
n = 3000

# データ生成
ad_spend = np.random.gamma(shape=2, scale=50, size=n)

impression = 20 * ad_spend + np.random.normal(0, 200, n)
click = 0.05 * impression + np.random.normal(0, 20, n)
stay_time = 2.0 * click + np.random.normal(0, 10, n)

sales = 3.0 * stay_time + np.random.normal(0, 30, n)

df = pd.DataFrame({
    "ad_spend": ad_spend,
    "impression": impression,
    "click": click,
    "stay_time": stay_time,
    "sales": sales
})

# 相関を見る
corr = df.corr()
print("======correlation analysis======")
print(corr["sales"].sort_values(ascending=False))

# 因果を意識したモデル
# ad_spend->impression->click->stay_time->sales
X_total = sm.add_constant(df["ad_spend"])
model_total = sm.OLS(df["sales"], X_total).fit()

print("======caused analysis1======")
print(model_total.params.drop("const"))

# 因果の設定を間違えたモデル
# ad_spend->impression->click->sales
# stay_time->sales
X_total = sm.add_constant(df[["ad_spend","stay_time"]])
model_total = sm.OLS(df["sales"], X_total).fit()

print("======caused analysis2======")
print(model_total.params.drop("const"))
