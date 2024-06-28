# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:40:08 2024

@author: Matheus Miyamoto
"""

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install factor_analyzer
!pip install sympy
!pip install scipy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install pingouin
!pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
import sympy as sy
import scipy as sp

#%%
paises_dados = pd.read_csv('Country-data.csv', sep=',', decimal='.')
#https://www.kaggle.com/datasets/vipulgohel/clustering-pca-assignment?resource=download

pais_pca = paises_dados
pais_pca = pais_pca.drop(columns=['country'])
print(paises_dados.info())
#%%
pais_pca = paises_dados[['child_mort','exports','health','imports','income','inflation','life_expec','total_fer','gdpp']]

pg.rcorr(pais_pca,method='pearson',upper='pval',decimals=4,
         pval_stars={0.01: '***',0.05: '**',0.10: '*'})

corr = pais_pca.corr()

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.4f}',
        colorscale='viridis'))

fig.update_layout(
    height = 600,
    width = 600,
    yaxis=dict(autorange="reversed"))

fig.show()

#%%Teste de Bartlett
bartlett, p_value = calculate_bartlett_sphericity(pais_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')
#%%PCA

fa = FactorAnalyzer(n_factors=9, method='principal', rotation=None).fit(pais_pca)
autovalores = fa.get_eigenvalues()[0]
#Seguindo o critério de Kaiser vamos utilizar apenas os 3 primeiros fatores
fa = FactorAnalyzer(n_factors=3, method='principal', rotation=None).fit(pais_pca)
autovalores_fatores = fa.get_factor_variance()
#%%
tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = pais_pca.columns

print(tabela_cargas)
#%%
comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = pais_pca.columns

print(tabela_comunalidades)
#%%
fatores = pd.DataFrame(fa.transform(pais_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]
paises_dados = pd.concat([paises_dados.reset_index(drop=True), fatores], axis=1)
#%% Scores

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = pais_pca.columns

#print(tabela_scores)

paises_dados['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    paises_dados['Ranking'] = paises_dados['Ranking'] + paises_dados[tabela_eigen.index[index]]*variancia
    
print(paises_dados)