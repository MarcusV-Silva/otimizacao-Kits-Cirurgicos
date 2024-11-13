import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def ler_dados():
  dt = pd.read_csv('Base_Filtrada.csv')
  return dt

#Procedimento de elaboracao dos kits
@st.cache_data
def faz_kits():
  dt = ler_dados()

  columns = ['DS_CIRURGIA', 'DS_PRODUTO_MESTRE', 'CD_AVISO_CIRURGIA','DS_ESPECIE']
  df = dt[columns].groupby(['DS_CIRURGIA', 'DS_PRODUTO_MESTRE','DS_ESPECIE']).agg(
      freq_m = pd.NamedAgg(column = 'CD_AVISO_CIRURGIA', aggfunc = pd.Series.nunique)).sort_values(by='freq_m', ascending = False).reset_index()


  freqs = dt[['DS_CIRURGIA','CD_AVISO_CIRURGIA']].groupby('DS_CIRURGIA').agg(pd.Series.nunique)
  freqs = dict(zip(freqs.index,freqs.CD_AVISO_CIRURGIA))

  columns = ['DS_CIRURGIA','NOVO_CD_PROD_MS','DS_PRODUTO_MESTRE','DS_CID','DS_PRODUTO','DS_UNIDADE_REFERENCIA','DS_ESPECIALID','CD_AVISO_CIRURGIA','QT_MOVIMENTACAO']
  aux = dt[columns].groupby(['DS_CIRURGIA','NOVO_CD_PROD_MS','DS_CID','DS_PRODUTO_MESTRE','DS_PRODUTO','DS_UNIDADE_REFERENCIA']).agg(
      min  = pd.NamedAgg(column='QT_MOVIMENTACAO',aggfunc='min'),
      max  = pd.NamedAgg(column='QT_MOVIMENTACAO',aggfunc='max'),
      mean  = pd.NamedAgg(column='QT_MOVIMENTACAO',aggfunc='mean'),
      median  = pd.NamedAgg(column='QT_MOVIMENTACAO',aggfunc='median'),
      std  = pd.NamedAgg(column='QT_MOVIMENTACAO',aggfunc='std'),
      freq_prod = pd.NamedAgg(column='CD_AVISO_CIRURGIA',aggfunc=pd.Series.nunique)).sort_values(by='freq_prod',ascending=False).reset_index()
  aux['freq_prod%'] = aux['freq_prod'] / aux['DS_CIRURGIA'].apply(lambda x: freqs[x])
  aux['std'].fillna(0,inplace=True)
  aux['qt'] = aux['median'] + round(aux['std'])
  aux['qt'] = aux.apply(lambda x: min(x['max'],x['qt']),axis=1)
  aux['key_value'] = aux.apply(lambda row: f"{row['NOVO_CD_PROD_MS']}->{row['DS_PRODUTO_MESTRE']}", axis=1)
  # Cruza tabelas para obter frequencias de mestres
  aux = pd.merge(left=aux,right=df,on=['DS_CIRURGIA',	'DS_PRODUTO_MESTRE'],how='left')
  aux['freq_m%'] = aux['freq_m'] / aux['DS_CIRURGIA'].apply(lambda x: freqs[x])
  aux.sort_values(by='freq_m%',inplace=True)

  return aux


#Funcao que mostra a quantidade de itens no primeiro kit
def kit1(a,b1,c,n1):
  aux = faz_kits()
  k1 = aux.loc[(a)&(b1)&(aux['freq_m%']>c)]
  k1.to_csv(f'kit_{n1}.csv')
  dt1 = pd.read_csv(f'kit_{n1}.csv')
  kit1 = dt1['key_value'].nunique()

  return st.subheader(f'{kit1}')

#Funcao que mostra a quantidade de itens no segundo kit
def kit2(a,b2,c,n2):
  aux = faz_kits()
  k2 = aux.loc[(a)&(b2)&(aux['freq_m%']>c)]
  k2.to_csv(f'kit_{n2}.csv')
  dt2 = pd.read_csv(f'kit_{n2}.csv')
  kit2 = dt2['key_value'].nunique()

  return st.subheader(f'{kit2}')


#Funcao que encontra o kit com os itens em comum
def inter(a,b1,b2,c,n1,n2):
  aux = faz_kits()

  k1 = aux.loc[(a)&(b1)&(aux['freq_m%']>c)]
  k1.to_csv(f'kit_{n1}.csv')
  dt1 = pd.read_csv(f'kit_{n1}.csv')
  kit1 = dt1['key_value'].nunique()

  k2 = aux.loc[(a)&(b2)&(aux['freq_m%']>c)]
  k2.to_csv(f'kit_{n2}.csv')
  dt2 = pd.read_csv(f'kit_{n2}.csv')
  kit2 = dt2['key_value'].nunique()

  inter = pd.merge(dt1, dt2, how = 'inner', on = 'key_value')
  inter.to_csv(f'kit_concat_{n1}_&_{n2}.csv')
  it = pd.read_csv(f'kit_concat_{n1}_&_{n2}.csv')
  intersection = it['key_value'].nunique()

  return st.subheader(f'{intersection}')

#Funcao que calcula a similaridade dos kits
def ComparaKits(a,b1,b2,c,n1,n2):
  aux = faz_kits()

  k1 = aux.loc[(a)&(b1)&(aux['freq_m%']>c)]
  k1.to_csv(f'kit_{n1}.csv')
  dt1 = pd.read_csv(f'kit_{n1}.csv')
  kit1 = dt1['key_value'].nunique()

  k2 = aux.loc[(a)&(b2)&(aux['freq_m%']>c)]
  k2.to_csv(f'kit_{n2}.csv')
  dt2 = pd.read_csv(f'kit_{n2}.csv')
  kit2 = dt2['key_value'].nunique()

  inter = pd.merge(dt1, dt2, how = 'inner', on = 'key_value')
  inter.to_csv(f'kit_concat_{n1}_&_{n2}.csv')
  it = pd.read_csv(f'kit_concat_{n1}_&_{n2}.csv')
  intersection = it['key_value'].nunique()

  if kit1 > kit2:
    result = intersection/kit1
    return st.subheader(f' {100*np.around(result,3)}%')
  else:
    result = intersection/kit2
    return st.subheader(f' {100*np.around(result,3)}%')

#Funcao que lista os itens em comum
def exibeinter(a,b1,b2,c,n1,n2):
  aux = faz_kits()

  k1 = aux.loc[(a)&(b1)&(aux['freq_m%']>c)]
  k1.to_csv(f'kit_{n1}.csv')
  dt1 = pd.read_csv(f'kit_{n1}.csv')
  kit1 = dt1['key_value'].nunique()

  k2 = aux.loc[(a)&(b2)&(aux['freq_m%']>c)]
  k2.to_csv(f'kit_{n2}.csv')
  dt2 = pd.read_csv(f'kit_{n2}.csv')
  kit2 = dt2['key_value'].nunique()

  inter = pd.merge(dt1, dt2, how = 'inner', on = 'key_value')
  inter.to_csv(f'kit_concat_{n1}_&_{n2}.csv')
  it = pd.read_csv(f'kit_concat_{n1}_&_{n2}.csv')
  intersection = it['key_value'].unique()

  return st.table(intersection)


aux = faz_kits()
c = 30/100 #corte feito nos kits

#Aqui comecam as funcoes para a exibicao na web dos resultados
st.subheader('Verificando a similaridade dos kits')

ci = st.text_input(
  'Insira o nome do procedimento com letras maiúsculas',
)

cid1 = st.text_input(
  'Insira o primeiro CID que será comparado',
)

cid2 = st.text_input(
  'Insira o segundo CID que será comparado',
)

if ci and cid1 and cid2:
  a = aux['DS_CIRURGIA']==ci
  b1 = aux['DS_CID']==cid1
  b2 = aux['DS_CID']==cid2
  n1 = cid1
  n2 = cid2

  st.subheader(f'Similaridade entre kits do procedimento {ci.capitalize()}')

  ComparaKits(a,b1,b2,c,n1,n2)

  st.subheader(f'Quantidade itens no kit de CID {cid1.capitalize()}')
  kit1(a,b1,c,n1)

  st.subheader(f'Quantidade itens no kit de CID {cid2.capitalize()}')
  kit2(a,b2,c,n2)

  st.subheader('Quantidade de itens na interseção dos kits')
  inter(a,b1,b2,c,n1,n2)

  st.subheader('Lista dos itens que fazem parte da interseção')
  exibeinter(a,b1,b2,c,n1,n2)
