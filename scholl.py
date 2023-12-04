import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('College_Data', index_col=0)
df.head()
df.info()
df['Private'].value_counts
#criando um scatterplot de Grad.Rate vs Room.Board
sns.lmplot(x= 'Room.Board', y= 'Grad.Rate', data=df, hue='Private')

#criando um scatterplot de F.Undergrad vs Outstate
sns.lmplot(x= 'F.Undergrad', y= 'Outstate', data=df, hue='Private')

g = sns.FacetGrid(df, hue='Private', height = 5)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.5)

#usando o pandas para demonstrar o grafico tambem
df[df['Private'] == 'Yes']['Outstate'].plot(kind='hist', alpha = 0.5)
df[df['Private'] == 'No']['Outstate'].plot(kind='hist', alpha = 0.5)

#criando um histograma para a coluna Grad.Rate
df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind='hist', alpha = 0.5, label='Private')
df[df['Private'] == 'No']['Grad.Rate'].plot(kind='hist', alpha = 0.5, label='Public')
plt.legend()

#mostre a escola que esta com uma taxa de graduação superior a 100%
df[df['Grad.Rate']> 100]

df['Grad.Rate']['Cazenovia College'] = 100

df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind='hist', alpha = 0.5, label='Private')
df[df['Private'] == 'No']['Grad.Rate'].plot(kind='hist', alpha = 0.5, label='Public')
plt.legend()

#CRIANDO CLUSTERS "KMEARS"
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

df_final = df.drop('Private', inplace=False, axis=1)

kmeans.fit(df_final)

#criando uma nova coluna para df chamado 'cluster', que é 1 para particular e 0 para publica

def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0
df['Cluster'] = df['Private'].apply(converter)

#matriz de confusao
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Cluster'], kmeans.labels_))
print(classification_report(df['Cluster'], kmeans.labels_))
