import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn.cluster import KMeans
from sklearn import datasets
import pylab

'''Opening file'''
df = pd.read_csv('data.csv', sep = ',', header = 0, index_col = 0)

#print(df, df.info())

'''Duplicates'''
#print(df.info(),df.drop_duplicates().info())


'''Missing values'''
#print(df.isna().sum())
df.dropna(inplace=True)
df.reset_index(level=0, inplace=True)


'''Connection rates by Content, Browser and ISP'''
VideoRate = df.groupby('#stream')['connected'].value_counts(normalize=True).unstack()
#VideoRate.plot.bar(stacked=True)
#plt.show()

BrowserRate = df.groupby('browser')['connected'].value_counts(normalize=True).unstack()
#BrowserRate.plot.bar(stacked=True)
#plt.show()

ISPRate = df.groupby('isp')['connected'].value_counts(normalize=True).unstack()
#ISPRate.plot.bar(stacked=True)
#plt.show()

''' Using ML on the whole data set
num_df = df.replace({'Arange':1,'BTP':2,'Datch Telecam':3,'Fro':4,'Olga':5,'EarthWolf':1,'Iron':2,'Swamp':3,'Vectrice':4})
#print(num_df)

data = num_df.values
#print(len(data[0,:]))

model=KMeans(n_clusters=3)
#adapter le modèle de données
model.fit(data)
#print(model)
print(model.labels_)
centroids = model.cluster_centers_
print(centroids)

colormap=np.array(['Red','green','blue',"yellow","black"])
plt.scatter(data[:,4],data[:,1],c=colormap[model.labels_],s=40)

plt.show()
'''


'''P2P vs CDN'''
df['ratio'] = df['p2p']/(df['p2p']+df['cdn'])


'''When p2p > 0'''
df_2 = df[df.ratio != 0]
dnld_isp = df_2.groupby('isp').mean()['ratio']
dnld_vid = df_2.groupby('#stream').mean()['ratio']
dnld_br = df_2.groupby('browser').mean()['ratio']
dnld = df_2.groupby(['#stream','isp']).mean()['ratio']
dnld_vid.plot.bar(stacked=True)
plt.show()
dnld_br.plot.bar(stacked=True)
plt.show()
dnld_isp.plot.bar(stacked=True)
plt.show()



'''When p2p = 0 but connected = True'''
#df['NoP2P'] = df.p2p.item() == 0 & df.connected.item() == True
#print(df['NoP2P'])

#VideoNoP2P = df.groupby('#stream')['connected'].value_counts(normalize=True).unstack()



