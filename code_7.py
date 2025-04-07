import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as sm_api
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# descriptive analytics
filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)
#boxplot for visualization (outliers)
df.boxplot(column = 'ARR_DELAY', by = 'OP_CARRIER_NAME')
plt.xticks(rotation = 90)
plt.show()
print(df.groupby('OP_CARRIER_NAME')['ARR_DELAY'].agg('mean'))
print(df.groupby('OP_CARRIER_NAME')['ARR_DELAY'].agg('median'))
#correlation for predictor variables
corr_matrix = df.corr(numeric_only=True)
corr_matrix = corr_matrix.round(2)
sns.heatmap(corr_matrix, annot=True, vmax=1, vmin =-1, cmap = 'icefire')
plt.show()


#Predictive Analytics 
airlines = ['American Airlines Inc.', 'Delta Air Lines Inc.', 'United Air Lines Inc.', 'Southwest Airlines Co.']
df = df.query("OP_CARRIER_NAME == @airlines")

model = sm_api.ols('ARR_DELAY ~ (DEP_DELAY)', data = df).fit()
anova_table = sm.stats.anova_lm(model,typ=2)
print(anova_table)

tukey_results = pairwise_tukeyhsd(endog=df ["ARR_DELAY"], groups=df["DEP_DELAY"], alpha =.05)
print(tukey_results)