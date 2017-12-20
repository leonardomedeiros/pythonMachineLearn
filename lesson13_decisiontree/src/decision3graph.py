import numpy as np
import pandas as pd
from sklearn import tree
from IPython.display import Image  
from sklearn.externals.six import StringIO  
import pydotplus

df = pd.read_excel("../xls/jogartenis.xls", sheetname=0)
print(df.head)
d = {'Sim': 1, 'Nao': 0}
df['JogarTenis'] = df['JogarTenis'].map(d)
d = {'Sol': 0, 'Nuvens': 1, 'Chuva': 2}
df['Aspecto'] = df['Aspecto'].map(d)
d = {'Quente': 0, 'Ameno': 1, 'Fresco': 2}
df['Temperatura'] = df['Temperatura'].map(d)
d = {'Normal': 0, 'Elevada': 1}
df['Humidade'] = df['Humidade'].map(d)
d = {'Fraco': 0, 'Forte': 1}
df['Vento'] = df['Vento'].map(d)

print(df.head())

features = list(df.columns[:6])
print(features)

y = df["JogarTenis"]
X = df[features]
clf = tree.DecisionTreeClassifier() 
clf = clf.fit(X,y)
#ypredicted = clf.predict(X)
print(clf.score(X,y))

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)  
graph = pydotplus.graphviz.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 

