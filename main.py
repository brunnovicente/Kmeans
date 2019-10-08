# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:19:46 2019

@author: brunn
"""
import pandas as pd
from kmeans import KMeans

dados = pd.read_csv('d:/basedados/mnist.csv')
X = dados.drop(['classe'], axis=1).values
y = dados['classe'].values

km = KMeans()
preditas = km.agrupar(X)