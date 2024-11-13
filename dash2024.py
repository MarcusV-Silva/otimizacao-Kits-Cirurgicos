import re
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import concurrent.futures
from functools import reduce
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

path = '/home/marcussilva/Documents/bases/'

st.set_page_config(layout="wide")

############################## Funções para ler a base de dados e carregar o modelo #################################

@st.cache_data
def readData():
  dt = pd.read_csv(path + 'novaBase.csv')
  return dt

@st.cache_data
def loadModel():
    dt = readData()
    dt.dropna(subset= 'DS_CID', inplace=True)

    columns = ['DS_CIRURGIA', 'DS_PRODUTO_MESTRE', 'CD_AVISO_CIRURGIA','DS_ESPECIE', 'DS_CLASSE']
    df = dt[columns].groupby(['DS_CIRURGIA', 'DS_PRODUTO_MESTRE','DS_ESPECIE', 'DS_CLASSE']).agg(
        freq_m = pd.NamedAgg(column = 'CD_AVISO_CIRURGIA', aggfunc = pd.Series.nunique)).sort_values(by='freq_m', ascending = False).reset_index()
    
    freqs = dt.groupby('DS_CIRURGIA')['CD_AVISO_CIRURGIA'].nunique()
    freqs_dict = freqs.to_dict()

    columns = ['DS_CIRURGIA', 'NOVO_CD_CIRURGIA','NOVO_CD_PRODUTO_MESTRE','DS_PRODUTO_MESTRE',
               'CD_CID','DS_CID','DS_PRODUTO','DS_UNIDADE_REFERENCIA','DS_ESPECIALID','CD_AVISO_CIRURGIA',
               'QT_MOVIMENTACAO', 'DS_CLASSE', 'ANO_MES_REALIZACAO']
    dt_selected = dt[columns]

    agg_funcs = {
        'QT_MOVIMENTACAO': ['min', 'max', 'mean', 'median', 'std'],
        'CD_AVISO_CIRURGIA': pd.Series.nunique
    }
    statistics = dt_selected.groupby(['DS_CIRURGIA', 'NOVO_CD_CIRURGIA','CD_AVISO_CIRURGIA',
                                      'NOVO_CD_PRODUTO_MESTRE','CD_CID','DS_CID','DS_PRODUTO_MESTRE',
                                      'DS_PRODUTO','DS_UNIDADE_REFERENCIA', 'DS_ESPECIALID', 'ANO_MES_REALIZACAO']).agg(agg_funcs)
    statistics.columns = ['min', 'max', 'mean', 'median', 'std', 'freq_prod']  
    statistics = statistics.sort_values(by='freq_prod', ascending=False).reset_index()
    statistics['freq_prod%'] = statistics['freq_prod'] / statistics['DS_CIRURGIA'].map(freqs_dict)
    statistics['std'].fillna(0, inplace=True)
    statistics['qt'] = statistics['median'] + round(statistics['std'])
    statistics['qt'] = statistics[['max', 'qt']].min(axis=1)

    aux = pd.merge(statistics, df, on=['DS_CIRURGIA', 'DS_PRODUTO_MESTRE'], how='left')
    aux['freq_m%'] = aux['freq_m'] / aux['DS_CIRURGIA'].map(freqs_dict)
    aux.sort_values(by='freq_m%', inplace=True)

    return aux

######################## Funções para verificar similaridade entre Produtos Mestre e Classes ########################

def similarityProd(df, cid):

    tmp = df.loc[(df['DS_CID'] == cid), ['DS_PRODUTO_MESTRE']]
    k1 = tmp['DS_PRODUTO_MESTRE'].nunique()

    k2 = df['DS_PRODUTO_MESTRE'].nunique()

    inter = pd.merge(df, tmp, how='inner', on= 'DS_PRODUTO_MESTRE')
    intersection = inter['DS_PRODUTO_MESTRE'].nunique()

    if k1 > k2:
        result = intersection/k1
        return result
    else:
        result = intersection/k2
        return result
    
def similarityClass(df, cid):

    tmp = df.loc[(df['DS_CID'] == cid), ['DS_CLASSE']]
    k1 = tmp['DS_CLASSE'].nunique()

    k2 = df['DS_CLASSE'].nunique()

    inter = pd.merge(df, tmp, how='inner', on= 'DS_CLASSE')
    intersection = inter['DS_CLASSE'].nunique()

    if k1 > k2:
        result = intersection/k1
        return result
    else:
        result = intersection/k2
        return result

################# Funções que geram os pontos, fazem o agrupamento e geram os dados para o K-Means ##################

def process_chunk(df, cid):
    x = similarityProd(df, cid)
    y = similarityClass(df, cid)
    return {"kitCid": cid, "prodSimilarity": x, "classSimilarity": y}

def kmeansData(df):
    columnCID = df['DS_CID'].drop_duplicates().to_numpy()
    
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, df, cid): cid for cid in columnCID}
        
        for future in concurrent.futures.as_completed(futures):
            cid = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                st.write(f"CID {cid} generated an exception: {exc}")

    return pd.DataFrame(results)

############### Funções para fazer as análises dos textos e retornar parte das quantidades dos itens ################

@st.cache_data
def genTuples(produto):

  formato = r'(\d+(\.\d+)?)\s*([A-Za-z]+)(?:/([A-Za-z]+))?'
  a = re.findall(formato, produto)

  output = [(valor, unidade1+'/'+unidade2 if unidade2 else unidade1)
          for valor, _, unidade1, unidade2 in a]
  return output

def genQuantities(df):
    df['DS_PRODUTO_MESTRE'] = df['DS_PRODUTO_MESTRE'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1.\2', x))

    df['qt_parcial'] = 1
    df['unidade_texto'] = df['DS_UNIDADE_REFERENCIA']

    for idx, row in df.iterrows():
        if row['DS_ESPECIE'] == 'MATERIAIS HOSPITALARES':
            continue

        produto = row['DS_PRODUTO_MESTRE']
        vet = genTuples(produto)

        if not vet:
            continue

        vet_filtrado = [tup for tup in vet if '/' not in tup[1]]
        if len(vet_filtrado) == 1:
            valor, unit = vet_filtrado[0]
            df.at[idx, 'qt_parcial'] = float(valor)
            df.at[idx, 'unidade_texto'] = unit

    df['Quant_final'] = df['qt_parcial'] * df['qt']

    df = df.iloc[:, [0, 1, 2, 3, 4, 6, 5]]

    return df

############################ Funções para definir os kits base,ou seja, os kits finais ##############################

@st.cache_data
def baseKit(df, list):
    i = 0
    dataframes = []
    v = list['kitCid'].drop_duplicates()
    v = v.to_numpy()

    while(i < len(v)):
        aux = v[i]

        dt = df.loc[(df['DS_CID'] == aux), ['DS_PRODUTO_MESTRE']]
        dataframes.append(dt)
        i += 1

    L = [pd.DataFrame(np.sort(x.values, axis=1), columns=x.columns).drop_duplicates() for x in dataframes]
    dfBase = reduce(lambda left, right: pd.merge(left,right), L)

    prev_kit = pd.merge(df, dfBase, how='inner', on = 'DS_PRODUTO_MESTRE')

    prev_kit = prev_kit.groupby(['DS_PRODUTO_MESTRE', 'DS_UNIDADE_REFERENCIA', 'DS_ESPECIE'])['qt'].min().reset_index()
    prev_kit['Qt'] = prev_kit['qt']

    kit = {'DS_PRODUTO_MESTRE': prev_kit['DS_PRODUTO_MESTRE'].to_list(),
           'qt': np.around(prev_kit['Qt'],1).to_list(),
           'DS_UNIDADE_REFERENCIA': prev_kit['DS_UNIDADE_REFERENCIA'].to_list(),
           'DS_ESPECIE': prev_kit['DS_ESPECIE'].to_list()}

    kit = pd.DataFrame(kit)

    kit = genQuantities(kit)

    return kit

@st.cache_data
def finalKit(df, cid):
    data = kmeansData(df)

    wcss = []
    distances = []

    for k in range(2,10):
        kmeans = KMeans(n_clusters=k-1, n_init='auto', random_state=42)
        kmeans.fit(data[['prodSimilarity','classSimilarity']])

        wcss.append(kmeans.inertia_)

    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss) - 1]

    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 -y1)**2 + (x2 - x1)**2)

        distances.append(numerator/denominator)

    numCluster = distances.index(max(distances)) + 2

    kmeans = KMeans(n_clusters=numCluster, n_init='auto', random_state=42)
    kmeans.fit(data[['prodSimilarity','classSimilarity']])

    data['cidGrupo'] = kmeans.labels_

    tmp1 = data.loc[(data['kitCid'] == cid)]
    tmp2 = tmp1['cidGrupo']

    list = data.loc[(data['cidGrupo'] == tmp2.iloc[0])].reset_index()

    res = baseKit(df, list)

    return res, list


################################# Funções de plot de alguns dos gráficos da interface ###############################

def barChartEsp(esp):
    aux = loadModel()

    anew = aux.loc[aux['DS_ESPECIALID'] == esp]
    val = anew.groupby(['DS_CIRURGIA', 'NOVO_CD_CIRURGIA']).nunique()[['CD_AVISO_CIRURGIA']].rename(columns={'CD_AVISO_CIRURGIA': 'counts'}).reset_index()

    limit = val.nlargest(10, 'counts')
    
    bar_chart = alt.Chart(limit).mark_bar().encode(
        x= alt.X('DS_CIRURGIA', title='Procedimentos Cirúrgicos'),
        y= alt.Y('counts', title='Quantidade'),
        tooltip=[alt.Tooltip('DS_CIRURGIA', title='Cirurgia'), alt.Tooltip('counts', title='Quantidade')]
    ).properties(
        title=f'Procedimentos mais frequentes'
    )
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    st.altair_chart(bar_chart, use_container_width=True)

def barChartCid(cir):

    anew = aux.loc[aux['DS_CIRURGIA'] == cir]
    val = anew.groupby(['DS_CID', 'CD_CID']).nunique()[['CD_AVISO_CIRURGIA']].rename(columns={'CD_AVISO_CIRURGIA': 'counts'}).reset_index()

    limit = val.nlargest(10, 'counts')

    bar_chart = alt.Chart(limit).mark_bar().encode(
        x=alt.X('DS_CID', title=f'Doenças '),
        y=alt.Y('counts', title='Quantidade'),
        tooltip=[alt.Tooltip('DS_CID', title='CID'), alt.Tooltip('counts', title='Quantidade')]
    ).properties(
        title=f'CIDs mais frequentes',
    )

    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    st.altair_chart(bar_chart, use_container_width=True)

def barChartCir(cid):

    anew = aux.loc[aux['DS_CID'] == cid]
    val = anew.groupby(['DS_CIRURGIA', 'NOVO_CD_CIRURGIA']).nunique()[['CD_AVISO_CIRURGIA']].rename(columns={'CD_AVISO_CIRURGIA': 'counts'}).reset_index()

    limit = val.nlargest(10, 'counts')

    bar_chart = alt.Chart(limit).mark_bar().encode(
        x= alt.X('DS_CIRURGIA', title='Procedimentos Cirúrgicos'),
        y= alt.Y('counts', title='Quantidade'),
        tooltip=[alt.Tooltip('DS_CIRURGIA', title='Procedimento'), alt.Tooltip('counts', title='Quantidade')]
    ).properties(
        title=f'Procedimentos cirúrgicos mais frequentes'
    )
    
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    st.altair_chart(bar_chart, use_container_width=True)

##################################### AQUI COMEÇA O LAYOUT DA INTERFACE #############################################

image = '/home/marcussilva/github/Scientific_Research/imagens/LOt.png'

with st.sidebar:
    st.sidebar.image(image, use_column_width=True)
    opcao = option_menu(
        'Menu',
        ('Visão Geral', 'Consulta por Kit'),
        icons=['eye', 'clipboard'], menu_icon='cast', default_index=0
    )

st.image(image, width=500, use_column_width=False)

################################# DESCRICAO DAS FUNCIONALIDADES DAS PÁGINAS #########################################

pag11 = '''
<p style="text-align:justify; font-size:20px;">
Nesta seção você poderá consultar quais são os procedimentos mais recorrentes das especialidades médicas disponíveis, 
bem como saber quais são as doenças mais frequentes relacionadas aos procedimentos da especialidade que você escolheu,
por fim, no último bloco você conseguirá saber quais são as intervenções cirúrgicas mais realizadas devido a uma doença 
a sua escolha, sendo que essa está dentro do grupo dos CIDs relacionados a consulta anterior.
'''

pag12 = '''
<p style="text-align:justify; font-size:20px;">
Aqui você poderá observar a frequência de cirurgias para cada especialidade ao longo do tempo, para observar cada uma 
individualmente basta clicar na barra correspondente, o tamanho da barra indica a quantidade de meses em que ocorreram
procedimentos dessa especialidade.
'''

pag21 = '''
<p style="text-align:justify; font-size:20px;">
Nesta página você poderá consultar a configuração dos Kits, como itens e suas respectivas quantidades, poderão ser 
consultados tanto Kits mais gerais como Kits baseados em um CID que você escolher.
'''

pag22 = '''
<p style="text-align:justify; font-size:20px;">
Aqui você pode observar a ocorrência desse procedimento ao longo dos meses.
'''

##################################### PRIMEIRA PÁGINA DA INTERFACE ##################################################

if opcao == 'Visão Geral':

    st.markdown('# Visão Geral')
    st.markdown('## Consulta de frequência de Procedimentos Cirúrgicos, CIDs e Princípios ativos')
    st.markdown(pag11, unsafe_allow_html=True)
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)

    a1, a2, a3 = st.columns([1, 1, 1])

    aux = loadModel()

    with a1:
        list = aux['DS_ESPECIALID'].unique()
        
        esp = st.selectbox(
            "Escolha uma especialidade",
            (list),
            index=None,
            placeholder= "Digite o nome da especialidade..."
        )
        if esp:
            barChartEsp(esp)
    
    tmp1 = aux['DS_ESPECIALID'] == esp
    l2 = aux.loc[tmp1]
    l2 = l2['DS_CIRURGIA'].unique()

    with a2:

        cir = st.selectbox(
            "Escolha um procedimento",
            (l2),
            index=None,
            placeholder="Digite o nome de um procedimento...",
            key='cid graph'
        )
        if cir:
            barChartCid(cir)

    tmp2 = aux['DS_CIRURGIA'] == cir
    l3 = aux.loc[tmp2]
    l3 = l3['DS_CID'].unique()

    with a3:

        cid = st.selectbox(
            "Escolha uma doença",
            (l3),
            index=None,
            placeholder="Digite o nome de uma doença...",
            key='prod graph'
        )
        if cid:
            barChartCir(cid)

##################################### PARTE DE BAIXO DA PRIMEIRA PÁGINA #############################################

    if esp and cir and cid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('## Frequência de Procedimentos para cada Especialidade ao longo do tempo')
        st.markdown(pag12, unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        aux = loadModel()

        aux['CD_AVISO_CIRURGIA'] = aux['CD_AVISO_CIRURGIA'].astype(int)
        aux['ANO_MES_REALIZACAO'] = pd.to_datetime(aux['ANO_MES_REALIZACAO'], format='%Y%m')

        d = aux.groupby(['DS_ESPECIALID', 'ANO_MES_REALIZACAO']).nunique()[['CD_AVISO_CIRURGIA']].rename(columns={'CD_AVISO_CIRURGIA': 'Freq'}).reset_index()
        
        esp = d['DS_ESPECIALID'].unique().tolist()
        
        base_colors =  [
            "#2E86C1",  
            "#F39C12",  
            "#27AE60",  
            "#8E44AD",  
            "#E74C3C",  
            "#F1C40F",  
            "#16A085",  
            "#2980B9",  
            "#C0392B",  
            "#D35400",  
            "#7F8C8D",  
            "#8E44AD",  
            "#1ABC9C", 
            "#3498DB",  
            "#9B59B6"   
        ]


        scale = alt.Scale(domain=esp, range=base_colors)
        
        color = alt.Color('DS_ESPECIALID:N', scale=scale, title= 'Especialidade')

        brush = alt.selection(type='interval' ,encodings=['x'])
        click = alt.selection_point(bind='scales', encodings=["color"])


        fig = (
            alt.Chart(d).mark_point().encode(
            x = alt.X('ANO_MES_REALIZACAO:T', title='Data (Mês e ano)'),
            y = alt.Y('Freq:Q', title='Frequência'),
            color = color
        )
        .properties(width = 800, height = 500)
        .transform_filter(click)
        .add_selection(brush)
        )

        bar = (alt.Chart(d).mark_bar().encode(
            x = alt.X('count()', title='Quantidade de meses em que cada procedimento ocorreu'),
            y = alt.Y('DS_ESPECIALID:N', title='Especialidade'),
            color = color
        ).properties(width = 700, height = 500)
        .add_selection(click)
        .transform_filter(brush)
        )

        chart = alt.hconcat(fig, bar).configure_legend(disable=True)
        st.altair_chart(chart, theme='streamlit',use_container_width=True)

##################################### SEGUNDA PÁGINA DA INTERFACE ##################################################

### PRÓXIMO PASSO: Colocar dois botões, com 'sim' e 'não', se sim vai fazer o kit sem CID, se não vai sumir com os botões e vai aparecer a caixa de seleção

elif opcao == 'Consulta por Kit':

    st.markdown('# Consulta de Kits Gerais e Kits baseados em CIDs')
    st.markdown('## Kit Geral e Kit Base com os CIDs relacionados')
    st.markdown(pag21, unsafe_allow_html=True)

    aux = loadModel()

    l1 = aux['DS_CIRURGIA'].unique()
    
    option = st.radio(
        'Escolha a forma como deseja que seu kit ser elaborado',
        ['***Kit Geral***', '***Kit com CID a sua escolha***']
    )

    if option == '***Kit Geral***':
        cir = st.selectbox(
            'Escolha um procedimento cirúrgico',
            (l1),
            index=None,
            placeholder= 'Digite o nome de um procedimento...',
            key= 'lista de cirurgias para consultar cid'
        )

        cid = cir

        tmp = aux['DS_CIRURGIA'] == cir

        df = aux.loc[(tmp)&(aux['freq_m%']>0.3), ['DS_CID', 'DS_CLASSE','DS_PRODUTO_MESTRE', 'qt', 'DS_UNIDADE_REFERENCIA', 'DS_ESPECIE']]

        Df = df.groupby(['DS_PRODUTO_MESTRE', 'DS_UNIDADE_REFERENCIA', 'DS_ESPECIE'])['qt'].min().reset_index()

        Df = genQuantities(Df)

        c1, c2 = st.columns([1, 1])

        if cir:
            colunms = ['DS_PRODUTO_MESTRE', 'DS_ESPECIE', 'Quant_final', 'unidade_texto']
            kit = Df[colunms]
            kit.columns = ['Princípio Ativo', 'Espécie', 'Quantidade', 'Unidade']

            df = df.groupby(['DS_CID'])['qt'].min().reset_index()
            df.rename(columns={'DS_CID': f'Doenças relacionadas ao procedimento {cir.capitalize()}'}, inplace=True)
            cids = df[f'Doenças relacionadas ao procedimento {cir.capitalize()}']

            with c1:
                st.write(kit)
            with c2:
                st.write(cids)
    else:
        b1, b2 = st.columns([1, 1])

        with b1:
            cir = st.selectbox(
                'Escolha um procedimento cirúrgico',
                (l1),
                index=None,
                placeholder= 'Digite o nome de um procedimento...',
                key= 'lista de cirurgias para consultar cid'
            )

        tmp = aux['DS_CIRURGIA'] == cir
        l2 = aux.loc[tmp]
        l2 = l2['DS_CID'].unique()

        with b2:
            #st.button("Reset", type="primary")
            cid = st.selectbox(
                'Escolha uma doença relacionada ao procedimento anterior',
                (l2),
                index=None,
                placeholder= 'Digite o nome do CID',
                key= 'Cid para fazer o kit'
            )

        st.markdown("<br><br>", unsafe_allow_html=True)

        df = aux.loc[(tmp)&(aux['freq_m%']>0.3), ['DS_CID', 'DS_CLASSE','DS_PRODUTO_MESTRE', 'qt', 'DS_UNIDADE_REFERENCIA', 'DS_ESPECIE']]

        c1, c2 = st.columns([1, 1])

        if cir and cid:
            kit, cids = finalKit(df, cid)
            
            colunms = ['DS_PRODUTO_MESTRE', 'DS_ESPECIE', 'Quant_final', 'unidade_texto']
            kit = kit[colunms]
            kit.columns = ['Princípio Ativo', 'Espécie', 'Quantidade', 'Unidade']

            colunms = ['kitCid']
            cids = cids[colunms]
            cids.columns = ['CIDs que possuem o mesmo kit base do CID pesquisado']

            with c1:
                st.write(kit)
            with c2:
                st.write(cids)

##################################### PARTE DE BAIXO DA SEGUNDA PÁGINA ##############################################

    if cir and cid:
        subheader = st.empty()
        subheader.markdown(f'# Histórico de ocorrência do procedimento {cir.capitalize()} ao longo do tempo')
        st.markdown(pag22, unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

    if cir and cid:
        aux['ANO_MES_REALIZACAO'] = pd.to_datetime(aux['ANO_MES_REALIZACAO'], format='%Y%m')
        d = aux.groupby(['DS_CIRURGIA', 'ANO_MES_REALIZACAO']).nunique()[['CD_AVISO_CIRURGIA']].rename(columns={'CD_AVISO_CIRURGIA': 'Freq'}).reset_index()
        d = d.loc[d['DS_CIRURGIA'] == cir]

        fig = (
            alt.Chart(d).mark_point().encode(
            alt.X('ANO_MES_REALIZACAO:T', title='Data (Mês e ano)', 
                    axis=alt.Axis(format= '%b %Y', tickCount='month', labelAngle=0)),
            alt.Y('Freq:Q', title='Frequência')
        )
        .properties(width = 800, height = 500)
        )

        st.altair_chart(fig, theme='streamlit',use_container_width=True)
