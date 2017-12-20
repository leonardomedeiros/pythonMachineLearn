import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_excel('../xls/cars.xls')
#normalizar os dados por uma escala
scale = StandardScaler()
X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']
X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())
print (X)
est = sm.OLS(y, X).fit()
print(est.summary())
print(y.groupby(df.Cylinder).mean())
