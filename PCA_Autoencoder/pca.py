# > %reset  #clear all variables
"""
@author: Srikanth Amudala | Deeksha
@desc: PCA Data Analysis using Wine Data set from sklearn
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import pandas

from sklearn.model_selection import train_test_split
sns.set()
# from keras.datasets import


wine = load_wine()
df_x = pandas.DataFrame(wine.data, columns=wine.feature_names)
df_y = pandas.DataFrame(wine.target)
x_train, y_train, x_test, y_test = train_test_split(df_x, df_y, test_size=0.2)

df = pd.DataFrame(x_train)
# normalize data
df = (df - df.mean()) / df.std()
# Displaying DataFrame columns.
df.columns
# Some basic information about each column in the DataFrame
df.info()

# bservations and variables
observations = list(df.index)
variables = list(df.columns)
# visualisation of the data using a box plot
sns.boxplot(data=df, orient="v", palette="Set2")
plt.figure(figsize=(100,100), num=1)
ax = sns.boxplot(data=df, orient="v", palette="Set2")
plt.xticks(rotation=60)
plt.title('Box Plot')
sns.pairplot(df)
plt.title('Pairwise relationships in a dataset.')

# Covariance
dfc = df - df.mean()  # centered data
plt.figure(figsize=(100,100), num=3)

corr = dfc.cov()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True


with sns.axes_style("white"):
    ax = sns.heatmap(corr, cmap='RdYlGn_r', mask=mask, square=True, annot=True)

sns.set(font_scale=0.6)
ax.tick_params(labelbottom=True, labeltop=False, rotation=60)
plt.title('Covariance matrix')

# Principal component analysis
pca = PCA()
pca.fit(df)
Z = pca.fit_transform(df)

plt.figure(4)
plt.scatter(Z[:, 0], Z[:, 1], c='r')
plt.xlabel('$Z_1$')
plt.ylabel('$Z_2$')
# sns.set(font_scale=0.7)

for label, x, y in zip(observations, Z[:, 0], Z[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.title("PCA by taking first two principal components")

# Eigenvectors
A = pca.components_
print("Eigen Vectors: ", A)
plt.figure(5)
plt.scatter(A[:, 0], A[:, 1], c='r')
plt.xlabel('$A_1$ Coefficient')
plt.ylabel('$A_2$ Coefficient')
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-2, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.title("PC Coefficients")
# plt.show()

plt.figure(6)
plt.scatter(A[:, 0], A[:, 1], marker='o', c=A[:, 2], s=A[:, 3] * 500,
            cmap=plt.get_cmap('Spectral'))
for label, x, y in zip(variables, A[:, 0], A[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(-20, 20),
                 textcoords='offset points', ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


# Explained Variance

# Eigenvalues
Lambda = pca.explained_variance_
print("Lambda: ", Lambda)
# Scree plot
plt.figure(7)
x = np.arange(len(Lambda)) + 1
plt.plot(x, Lambda, 'ro-', lw=2)
plt.xticks(x, ["Comp." + str(i) for i in x], rotation=60)

plt.ylabel('Explained variance')
# Explained variance
ell = pca.explained_variance_ratio_

# Explained Variance using bar graph
plt.figure(8)
ind = np.arange(len(ell))
plt.bar(ind, ell, align='center', alpha=0.5)
plt.plot(np.cumsum(ell))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
# plt.show()

# Biplot
# 0,1 denote PC1 and PC2; change values for other PCs
A1 = A[0]
A2 = A[1]
Z1 = Z[:, 0]
Z2 = Z[:, 1]

plt.figure(9)
for i in range(len(A1)):
    # arrows project features as vectors onto PC axes
    plt.arrow(0, 0, A1[i] * max(Z1), A2[i] * max(Z2),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(A1[i] * max(Z1) * 1.2, A2[i] * max(Z2) * 1.2, variables[i], color='black')

for i in range(len(Z1)):
    # circles project documents (ie rows from csv) as points onto PC axes
    plt.scatter(Z1[i], Z2[i], c='g', marker='o')
    # plt.text(Z1[i] * 1.2, Z2[i] * 1.2, observations[i], color='b')

plt.title("2D Biplot for PC1 & PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
# plt.show()

plt.figure(num=10)
comps = pd.DataFrame(A, columns=variables)
sns.heatmap(comps, cmap='RdYlGn_r', linewidths=1, annot=True,
            cbar=True, square=True)
sns.set(font_scale=0.7)
ax.tick_params(labelbottom=False, labeltop=True, rotation=60)
plt.title('Principal components')

important_components = [(i/Lambda.sum())*100 for i in Lambda]
for i, j in enumerate(important_components):
    print("Explained Variance for Component "+str(i)+" : ", round(j)," %")

plt.show()
