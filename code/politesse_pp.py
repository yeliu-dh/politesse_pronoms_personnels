# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 09:52:42 2024

@author: Ye
"""




###IMPORT
import csv
import pandas as pd
import matplotlib as plt
import numpy as np
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.decomposition import FactorAnalysis

import statsmodels.api as sm
import statsmodels.formula.api as smf


def cronbach_alpha(df):
    item_scores = df.values
    var_sum = np.sum(np.var(item_scores, axis=0, ddof=1))
    total_var = np.var(np.sum(item_scores, axis=1), ddof=1)
    n_items = item_scores.shape[1]
    return (n_items / (n_items - 1)) * (1 - var_sum / total_var)



####CODE 
####merge : politesse_zsc
listings_unique_poli=pd.read_csv('..\\data\\listings_unique_labeled.csv',sep=',', on_bad_lines='skip',encoding='utf-8')
listings_poli=pd.read_csv('..\\data\\listings_poli.csv',sep=',', on_bad_lines='skip',encoding='utf-8')

listings_poli=listings_poli.drop(columns=['bienvaillent', 'gentil', 'attentif', 'inclusive', 'courtois','tolerant', 'respectueux', 'emotional', 'behavioral', 'politesse'])
listings_poli=listings_poli.drop(columns=['poli'])
print(listings_poli.columns)


listings_poli=listings_poli.merge(listings_unique_poli[['text', 'bienveillant', 'poli', 'sincère', 'attentif', 'inclusive',
       'courtois', 'patient', 'respectueux', 'compréhensif', 'formel',
       'discret']], left_on='host_about',right_on='text',how='left')
listings_poli.drop(columns='text',inplace=True)
print(listings_poli.columns)
# listings_poli.to_csv('..\\data\\listings_poli_2.csv', index=False)


###merge : emotions
listings_unique_sentiments=pd.read_csv('..\\data\\listings_unique_sentiments.csv',sep=',', on_bad_lines='skip',encoding='utf-8')
listings_poli=listings_poli.merge(listings_unique_sentiments[['host_about','sentiment', 'polarity', 'subjectivity']], left_on='host_about',right_on='host_about',how='left')
print(listings_poli.columns)
#listings_poli.to_csv('..\\data\\listings_poli_2.csv', index=False)



# print(listings_poli.sentiment.value_counts())
# print(listings_poli.sentiment.isna().value_counts())
#print(listings_poli.polarity.describe())#mean         0.195668 大于0积极，小于0消极
#print(listings_poli.subjectivity.describe())#mean         0.345507



data=listings_poli[['bienveillant', 'poli', 'sincère', 'attentif', 'inclusive','courtois', 'patient', 'respectueux', 'compréhensif', 'formel','discret']]

# #alpha:
# #print(data.isna().value_counts())#2
# data=data.fillna(0)
# #a=cronbach_alpha(data)
# #print(a)#0.892769056061534

# labels=data.columns
# print(labels)
# fa = FactorAnalysis(n_components=1)
# factor_scores = fa.fit_transform(data)
# print(f"Factor Loadings: {fa.components_}")

# emotional_labels = ['bienveillant', 'inclusive', 'sincère','compréhensif']
# behavioral_labels = ['patient', 'attentif','courtois']
# cognitive_labels = ['respectueux','poli','formel']
# labels_valide=['bienveillant', 'inclusive', 'sincère','compréhensif','patient', 'attentif','courtois']


# alpha_e=cronbach_alpha(data[emotional_labels])
# print(alpha_e)
# alpha_b=cronbach_alpha(data[behavioral_labels])
# print(alpha_b)
# alpha_c=cronbach_alpha(data[cognitive_labels])
# print(alpha_c)


def weighted_average(labels, factor_loadings, df):
    # 计算加权平均
    weighted_sum = sum(df[label] * factor_loadings[i] for i,label in enumerate(labels))
    total_weight = sum(factor_loadings[i] for i, label in enumerate(labels))
    return weighted_sum / total_weight

# # 计算每个维度的加权平均
# data['emotionnel'] = data.apply(lambda row: weighted_average(emotional_labels, factor_loadings, row), axis=1)
# data['comportemental'] = data.apply(lambda row: weighted_average(behavioral_labels, factor_loadings, row), axis=1)
# data['politesse'] = data.apply(lambda row: weighted_average(labels_valide, factor_loadings, row), axis=1)


#贴回listings_poli:
# #合并计算结果：
# data_=data[['emotionnel','comportemental','politesse']]
# listings_poli_=pd.concat([listings_poli,data_],axis=1)
# print(listings_poli_.columns)
#listings_poli_.to_csv('..\\data\\listings_poli_2.csv', index=False)






############pearson :
# data=listings_poli[['bienveillant', 'sincère', 'attentif','inclusive', 'courtois', 'patient', 'respectueux', 'compréhensif',
# 'formel', 'discret', 'emotionnel', 'comportemental', 'politesse']]

# data_e=data[['bienveillant', 'sincère','inclusive', 'compréhensif', 'emotionnel', 'politesse']]

# data_c=data[['attentif','courtois', 'patient','comportemental', 'politesse']]




# corr_matrix = data_e.corr(method='pearson')
# print(corr_matrix)
# plt.figure(figsize=(8,6))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
# plt.title("Pearson Correlation Matrix of Emotional Labels")
# #plt.title("Pearson Correlation Matrix of Comportemental Labels")
# plt.tight_layout()
# plt.savefig('..\\tal\\pearson_correlation_matrix_7W4_emotionnel.jpg', dpi=300)
# # 观察是否有标签与其他标签的相关系数显著低于0.5。





#####OLS
listings_poli=pd.read_csv('..\\data\\listings_poli_2.csv',sep=',', on_bad_lines='skip',encoding='utf-8')
#print(listings_poli.columns)

# # # #listings_poli=listings_poli_
# model=smf.ols('subjectivity~P1_rel+P2_rel+P3_rel', data=listings_poli).fit()
# print(model.summary())

# with open('..\\tal\\comportemental-P123_rel-lm_2.txt', 'w') as f:
#        f.write(model.summary().as_text())
       


# model=smf.ols('subjectivity+polarity', data=listings_poli).fit()
# print(model.summary())

#with open('..\\tal\\politesse-subjectivity_polarity-lm_2.txt', 'w') as f:
 #       f.write(model.summary().as_text())
 



# data=listings_poli[listings_poli['host_class_']!='Unknown']

# model=smf.ols('politesse~P1_rel+host_class_+P1_rel*host_class_', data=data).fit()
# print(model.summary())

# with open('..\\tal\\politesse-P1Xhost_class-lm_2.txt','w') as f:
#       f.write(model.summary().as_text())
       
ss
























