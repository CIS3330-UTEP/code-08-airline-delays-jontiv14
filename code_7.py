import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Descriptive Analytics
filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)
print(df.describe)


# query = "Identification_Make in ['Toyota', 'Mazda', 'Honda']"
# df = df.query(query)

# model = sm_api.ols('Dimensions_Length ~ C(Identification_Make)', data = df).fit()
# anova_table = sm.stats.anova_lm(model,typ=2)
# print(anova_table)

# tukey_results = pairwise_tukeyhsd(endog=df ["Dimensions_Length"], groups=df["Identification_Make"], alpha =.05)
# print(tukey_results)
# #ARR_DELAY is the column name that should be used as dependent variable (Y).