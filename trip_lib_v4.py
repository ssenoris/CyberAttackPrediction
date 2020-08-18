import pandas as pd
import datetime
import seaborn as sns
import numpy as np
from collections import deque
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
import string
import pickle
from termcolor import colored
from sklearn.metrics import mean_squared_error

# GLOBAL_TRAIN_COLUMNS = ['puerto_destino','escaner', 'len_min', 'len_max','n_paq_mean','n_paq_std','n_paq_min','n_paq_max','n_paquetes','Length','puerto_caliente']
GLOBAL_TRAIN_COLUMNS = ['puerto_destino','escaner','n_paquetes','puerto_caliente','Length']

#generacion del modelo para guardarlo y cargarlo rápido con el método prediccion
#hay que introducir los hiperparametros optimos del modelo
#las columnas que van a ir en el modelo ya están especificadas
#se ha quitado len_mean y len_std por tener alta correlacion con el target

def modelo_rf_op(X):
#     print(X.columns)
    x = GLOBAL_TRAIN_COLUMNS
    X_train, X_test, y_train, y_test = train_test_split(X[x], X['ataque'], test_size=0.3, random_state=42)
    
    numerical = X_train.select_dtypes(exclude=["category",'object']).columns
    categorical = X_train.select_dtypes(include=["object"]).columns
    t = [('cat', OneHotEncoder(), categorical), ('num', StandardScaler(), numerical)]
    transformer = ColumnTransformer(transformers=t)
    X_train_transformed = transformer.fit_transform(X_train)
    
    filename_trans = 'x_transform_OP.sav'
    
    pickle.dump(transformer, open(filename_trans, 'wb'))
#     print(X_test)
    X_test_transformed = transformer.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=95, min_samples_split = 10, min_samples_leaf = 2, max_features = 'auto', bootstrap = False,max_depth=4, random_state=0)
    rf.fit(X_train_transformed, y_train)
    filename = 'rf_op_model_OP.sav'
    pickle.dump(rf, open(filename, 'wb'))
    fitted_labels = rf.predict(X_test_transformed)
    score = accuracy_score( fitted_labels,y_test)
    print('max_depth = ',4,' score:', score)
def corrheat_triangular(df):
    sns.set(style="white")
    
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 8))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap,annot=True, vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})



def prediccion_(url):
    loaded_model = pickle.load(open('rf_op_model_OP.sav', 'rb'))
# loaded_transform = pickle.load(open('x_transform.sav', 'rb'))
# [trafico_normal[0]]
    X_test = x_test([url])
    
    y_pred = pd.Series(loaded_model.predict(X_test))
    valor = y_pred.value_counts(ascending = False, normalize = True)*100
    if 1 in valor.index and valor[1] <50:
#         print('Ataque')
        print(colored('Riesgo de ATAQUE', 'green'),round(valor[1],1),'%')
    elif 1 in valor.index and valor[1] >= 50:
        print(colored('Riesgo de ATAQUE', 'red'),round(valor[1],1),'%')
    else:
#         print('Tráfico normal')
        print(colored('TRÁFICO NORMAL', 'green'))

def prediccion(url):
    loaded_model = pickle.load(open('rf_op_model_OP.sav', 'rb'))
# loaded_transform = pickle.load(open('x_transform.sav', 'rb'))
# [trafico_normal[0]]
    X_test = x_test([url])
    
    y_pred = pd.Series(loaded_model.predict(X_test))
    valor = y_pred.value_counts(ascending = False, normalize = True)*100
    if 1 in valor.index and valor[1] <50:
#         print('Ataque')
        print(colored('Riesgo de ATAQUE', 'green'),round(valor[1],1),'%')
    elif 1 in valor.index and valor[1] >= 50:
        print(colored('Riesgo de ATAQUE', 'red'),round(valor[1],1),'%')
    else:
#         print('Tráfico normal')
        print(colored('TRÁFICO NORMAL', 'green'))
    
#     fig = detecta_ventana_lista_plot([url])
    return detecta_ventana_lista_plot([url]),procesado_no_supervisado([url])

def procesado_no_supervisado(list_url):
    dfs = []
    for i in list_url:
        #ABRO FICHERO
        
        file = pd.read_csv(i,encoding='latin1',sep = ';',error_bad_lines=False)
        if 'Time' not in file.columns:
            file = pd.read_csv(i,encoding='latin1',sep = ',',error_bad_lines=False)
        tsec = []
        t_uni = []
        con = ''
        #ESTADISTICAS CON LA FUNCION LENGTH
        len_mean = []
        len_std = []
        len_min = []
        len_max = []
        estadistica = file.Length.describe()
        for i in range(file.shape[0]):
            len_mean.append(estadistica[1])
            len_std.append(estadistica[2])
            len_min.append(estadistica[3])
            len_max.append(estadistica[7])
        file['len_mean'] = len_mean
        file['len_std'] = len_std
        file['len_min'] = len_min
        file['len_max'] = len_max
            
            
        #PONGO CORRECTAMENTE LA COLUMNA TIEMPO
        if type(file['Time'][0]) != str:
            file['Time'] = file['Time'].astype(str)
        for i in file['Time']:
            spliteo = i.split('.')
            con = spliteo[0]+'.'+spliteo[1]
            tsec.append(con)
            t_uni.append(spliteo[0])
        
        file['tiempo_dec(s)']=tsec
        file['tiempo_dec(s)'] = file['tiempo_dec(s)'].astype(float)
        file['t_uni']=t_uni
        file['t_uni'] = file['t_uni'].astype(float)
        
        #CALCULO N PAQUETES EN EL ULTIMO SEGUNDO
        q = deque()
        max_dif = 1
        #promedio del tiempo entre paquetes en las ventanas
        promedio_t_ultimo_s =[]
        promedio= 0
        n_paquetes = []
        ocurrencias = 0
        max_len_q = 0
        
        for i in file['tiempo_dec(s)']:
            q.append(i)
            elimina_valores(q,dif_max =max_dif)
            max_len_q = max(max_len_q, len(q))
            promedio = max_dif/len(q)
            n_paquetes.append(len(q))
            promedio_t_ultimo_s.append(promedio)
            
        file['n_paquetes'] = n_paquetes
        file['n_paquetes'].astype(float)
#         print(file['n_paquetes'])
        
        dfs.append(file)
        #CREO COLUMNA PARA INDICAR SI N PAQUETES ES > 300 PAQUETES/s
    df_con_indica_n_paquetes = []
    for df in dfs:
        alto_n_paquetes = [] 
        for i in df.n_paquetes:
            if i <300:
                alto_n_paquetes.append(0)
            else:
                alto_n_paquetes.append(1)
        df['alto_n_paquetes'] = alto_n_paquetes
        df_con_indica_n_paquetes.append(df)
    
    dfs_ataque = []
    
    #CREO COLUMNA PARA INDICAR SI LOS DATOS VAN A UN PUERTO CALIENTE 21 O 22
    dfs_pcaliente = []
    for df in df_con_indica_n_paquetes:
        puerto_caliente = []    
        for fila in range(df.shape[0]):
            if type(df['Info'][fila]) != str:
                df['Info'] = df['Info'].astype(str)
            if (' 21 ' in df.Info[fila]) or (' 22 ' in df.Info[fila]):
    #             print(file.Info[fila],'--tenemos 21 o 22')
                puerto_caliente.append(1)
            else:
                puerto_caliente.append(0)
        df['puerto_caliente'] = puerto_caliente
        dfs_pcaliente.append(df)
  
        
        
        #CREO COLUMNA INDICANDO PUERTO DESTINO
    df_total1 = columna_puerto_destino(dfs_pcaliente)
    
    df_total = escaner_puertos(df_total1)
    cont = 0
    for df in df_total:
        xx = df.puerto_destino.value_counts(ascending = False)[df.puerto_destino.value_counts(ascending = False).index != 0].index[1:4]
        yy = df.puerto_destino.value_counts(ascending = False)[df.puerto_destino.value_counts(ascending = False).index != 0][1:4]

        fig, axes = plt.subplots(1, 1, sharey=True, figsize=(6, 4))
        sns.set_color_codes("pastel")
        sns.barplot(x=xx, y=yy, data=df,label="numero de peticiones", color="b")
        sns.despine(left=True, bottom=True)
        plt.legend(ncol=2, loc='best', frameon=True)
        plt.xlabel('puerto destino')
        plt.ylabel('nº de peticiones')
        plt.savefig(list_url[cont]+'_puertos_mas_frecuentes.png')
        plt.show()
        plt.close()
        cont+=1
    #CREO COLUMNA PARA CONTAR POR CADA IP SI HAY IPS DESTINO QUE 
    df_total_concatenados = pd.concat(df_total)       
        
def detecta_ventana_lista_plot(lista_url, encoding='latin1', origen ='10.0.2.4', destino= '10.0.2.6'):
    
    for url in lista_url:
        file = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
        if 'Time' not in file.columns:
            file = pd.read_csv(url,encoding='latin1',sep = ',',error_bad_lines=False)
        #     file =file.iloc[900:,].reset_index()
        #     file = file[(file.Source == origen)&(file.Destination == destino)].reset_index()
        tsec = []
        con = ''
        if type(file['Time'][0]) != str:
            file['Time'] = file['Time'].astype(str)

        for i in file['Time']:
            spliteo = i.split('.')
        #     print(spliteo)
            con = spliteo[0]+'.'+spliteo[1]
        #     print(con)
            tsec.append(con)

        file['tiempo_dec(s)']=tsec
        #     print('tsec')
#         print(file['tiempo_dec(s)'])
        file['tiempo_dec(s)'] = file['tiempo_dec(s)'].astype(float)
        q = deque()
        max_dif = 1
        #promedio del tiempo entre paquetes en las ventanas
        promedio_t_ultimo_s =[]
        promedio= 0
        n_paquetes = []
        ocurrencias = 0
        max_len_q = 0
        
        for i in file['tiempo_dec(s)']:
            q.append(i)
        #         print('q:',q)
            elimina_valores(q,dif_max =max_dif)
        #        print(len(q))
            max_len_q = max(max_len_q, len(q))
            promedio = max_dif/len(q)
            n_paquetes.append(len(q))
        #        print('paquetes: ',n_paquetes)

            promedio_t_ultimo_s.append(promedio)
        #        print(promedio)
#             if promedio < max_dif/1000:
#         #             print('promedio: ',promedio)
#                 ocurrencias+=1
#                 if ocurrencias ==1:
#                     print('AVISO. Posible ataque')
        #             elif ocurrencias > 1:
        #                 print('¡¡¡Esto es una ataque!!!')
        #                 break
        print(url,' Maximo número de paquetes:', max_len_q)
#         print('Max n paquetes:', max(n_paquetes))
        sns.set(style="darkgrid")

        # Plot the responses for different events and regions
        df=pd.DataFrame()
        df['peticiones_ultimo_segundo'] = n_paquetes
        df['linea_temporal'] = [i for i in range(len(n_paquetes))]
#            sns.lineplot(x='peticiones_ultimo_segundo', y='linea_temporal', data=df)
        fig, axes = plt.subplots(1, 1, sharey=True, figsize=(8, 6))
        sns.lineplot(y=n_paquetes, x=[i for i in range(len(n_paquetes))])
        plt.xlabel('número de peticiones')
        plt.ylabel('peticiones/s')
        plt.savefig(url+'_peticiones_s.png')
        plt.show()

        

        return fig

def x_test(list_trafico):
    
    df = procesado_test(list_trafico)
    data_complete_2=source_destination(df)
    loaded_transform = pickle.load(open('x_transform_OP.sav', 'rb'))
    x_test_transdormed = loaded_transform.transform(data_complete_2)
    return x_test_transdormed



def procesado_test(list_url):
    dfs = []
    for i in list_url:
        #ABRO FICHERO
        
        file = pd.read_csv(i,encoding='latin1',sep = ';',error_bad_lines=False)
        if 'Time' not in file.columns:
            file = pd.read_csv(i,encoding='latin1',sep = ',',error_bad_lines=False)
        tsec = []
        t_uni = []
        con = ''
        #ESTADISTICAS CON LA FUNCION LENGTH
        len_mean = []
        len_std = []
        len_min = []
        len_max = []
        estadistica = file.Length.describe()
        for i in range(file.shape[0]):
            len_mean.append(estadistica[1])
            len_std.append(estadistica[2])
            len_min.append(estadistica[3])
            len_max.append(estadistica[7])
        file['len_mean'] = len_mean
        file['len_std'] = len_std
        file['len_min'] = len_min
        file['len_max'] = len_max
            
            
        #PONGO CORRECTAMENTE LA COLUMNA TIEMPO
        if type(file['Time'][0]) != str:
            file['Time'] = file['Time'].astype(str)
        for i in file['Time']:
            spliteo = i.split('.')
            con = spliteo[0]+'.'+spliteo[1]
            tsec.append(con)
            t_uni.append(spliteo[0])
        
        file['tiempo_dec(s)']=tsec
        file['tiempo_dec(s)'] = file['tiempo_dec(s)'].astype(float)
        file['t_uni']=t_uni
        file['t_uni'] = file['t_uni'].astype(float)
        
        #CALCULO N PAQUETES EN EL ULTIMO SEGUNDO
        q = deque()
        max_dif = 1
        #promedio del tiempo entre paquetes en las ventanas
        promedio_t_ultimo_s =[]
        promedio= 0
        n_paquetes = []
        ocurrencias = 0
        max_len_q = 0
        
        for i in file['tiempo_dec(s)']:
            q.append(i)
            elimina_valores(q,dif_max =max_dif)
            max_len_q = max(max_len_q, len(q))
            promedio = max_dif/len(q)
            n_paquetes.append(len(q))
            promedio_t_ultimo_s.append(promedio)
            
        file['n_paquetes'] = n_paquetes
        file['n_paquetes'].astype(float)
#         print(file['n_paquetes'])
        
                        #ESTADISTICAS CON LA FUNCION n paquetes
        n_paq_mean = []
        n_paq_std = []
        n_paq_min = []
        n_paq_max = []
        estadistica2 = file.n_paquetes.describe()
        for i in range(file.shape[0]):
            n_paq_mean.append(estadistica2[1])
            n_paq_std.append(estadistica2[2])
            n_paq_min.append(estadistica2[3])
            n_paq_max.append(estadistica2[7])
        file['n_paq_mean'] =  n_paq_mean
        file['n_paq_std'] =  n_paq_std
        file['n_paq_min'] =  n_paq_min
        file['n_paq_max'] =  n_paq_max
        
        dfs.append(file)
        #CREO COLUMNA PARA INDICAR SI N PAQUETES ES > 300 PAQUETES/s
#     df_con_indica_n_paquetes = []
#     for df in dfs:
#         alto_n_paquetes = [] 
#         for i in df.n_paquetes:
#             if i <300:
#                 alto_n_paquetes.append(0)
#             else:
#                 alto_n_paquetes.append(1)
#         df['alto_n_paquetes'] = alto_n_paquetes
#         df_con_indica_n_paquetes.append(df)
    
#     dfs_ataque = []
    
    #CREO COLUMNA PARA INDICAR SI LOS DATOS VAN A UN PUERTO CALIENTE 21 O 22
    dfs_pcaliente = []
    for df in dfs:
        puerto_caliente = []    
        for fila in range(df.shape[0]):
            if type(df['Info'][fila]) != str:
                df['Info'] = df['Info'].astype(str)
            if (' 21 ' in df.Info[fila]) or (' 22 ' in df.Info[fila]):
    #             print(file.Info[fila],'--tenemos 21 o 22')
                puerto_caliente.append(1)
            else:
                puerto_caliente.append(0)
        df['puerto_caliente'] = puerto_caliente
        df['puerto_caliente'] = df['puerto_caliente'].astype(str)
        dfs_pcaliente.append(df)
        #CREO COLUMNA INDICANDO PUERTO DESTINO
    df_total1 = columna_puerto_destino(dfs_pcaliente)
    
    df_total = escaner_puertos(df_total1)
#     print(df_total[0].columns)
    
    #CREO COLUMNA PARA CONTAR POR CADA IP SI HAY IPS DESTINO QUE 
    df_total_concatenados = pd.concat(df_total)
#     sns.lineplot(y=n_paquetes, x=[i for i in range(len(n_paquetes))])
#     plt.xlabel('ocurrencias en el ultimo segundo')
#     plt.ylabel('peticiones/s')
#     df_total[['tiempo_dec(s)','n_paquetes','ataque']]
    return df_total_concatenados[['escaner','len_mean','len_std', 'len_min', 'len_max','puerto_destino','Protocol','Source','Destination','n_paq_mean','n_paq_std','n_paq_min','n_paq_max','n_paquetes','Length','puerto_caliente']]

def preprocesado(trafico_ataques, trafico_normal):
    df_ataques = procesado_datos(trafico_ataques,lista_ataques = True)
    df_normal = procesado_datos(trafico_normal,lista_ataques = False)
    data_complete = pd.concat([df_ataques,df_normal]).reset_index(drop = True)
    data_complete_2=source_destination(data_complete)
    np.random.seed(seed=42)
    data_complete_random = data_complete_2.loc[np.random.permutation(len(data_complete))]
    
    return data_complete_random
def escaner_puertos(lista_dfs): 
    #1 si se produce escaneo de puertos
    #0 si no lo hace
    lista_dfs2 = []
    for df in lista_dfs:
        escaner_puertos = []
#         print(type(df))
        for valor in df.Destination.unique():
            df2 = df[df.Destination == valor]
#             print('valor:', valor, 'unique:', df2.puerto_destino.unique())
            escaner_puertos.append(len(df2.puerto_destino.unique()))
#             print(escaner_puertos)
#             longitud = len(df2.puerto_destino.unique())
        if max(escaner_puertos) >5:
            df['escaner'] = [1 for i in range(df.shape[0])]
        else:
            df['escaner'] = [0 for i in range(df.shape[0])]
        df['escaner'] = df['escaner'].astype(str)
        lista_dfs2.append(df)
    return lista_dfs2

def columna_puerto_destino(df_lista):
    lista_dfs = []  
    for df in df_lista:
#         print(df)
        puerto_destino = []
        for fila in range(df.shape[0]):
            if type(df['Info'][fila]) != str:
                df['Info'] = df['Info'].astype(str)
#            print(df.Info[fila])
            try:
                puerto = df.Info[fila].strip().split('>')[1].split()[0]
                puerto_float = float(puerto)
                puerto_destino.append(puerto)
            except:
                puerto = df.Info[fila]
                puerto_destino.append(0)
#                 puerto_destino.append(puerto)

        df['puerto_destino'] = puerto_destino
        df['puerto_destino'] = df['puerto_destino'].astype(int)
        
#         df['puerto_destino'] = df['puerto_destino'].astype('int64')
    
        lista_dfs.append(df)
    return lista_dfs



def source_destination(df):
    source = []
    for i in df.Source:
        if ':' in i:
            source.append(str(0))
        else:
            source.append(i)
    
    destination = []
    for i in df.Destination:
        if ':' in i:
            destination.append(str(0))
        else:
            destination.append(i)
    df['source'] = source
    df['destination'] = destination
    
    return df
            

def procesado_datos(list_url,lista_ataques = True):
    dfs = []
    for i in list_url:
        #ABRO FICHERO
        
        file = pd.read_csv(i,encoding='latin1',sep = ';',error_bad_lines=False)
        if 'Time' not in file.columns:
            file = pd.read_csv(i,encoding='latin1',sep = ',',error_bad_lines=False)
        tsec = []
        t_uni = []
        con = ''
        #ESTADISTICAS CON LA FUNCION LENGTH
        len_mean = []
        len_std = []
        len_min = []
        len_max = []
        estadistica = file.Length.describe()
        for i in range(file.shape[0]):
            len_mean.append(estadistica[1])
            len_std.append(estadistica[2])
            len_min.append(estadistica[3])
            len_max.append(estadistica[7])
        file['len_mean'] = len_mean
        file['len_std'] = len_std
        file['len_min'] = len_min
        file['len_max'] = len_max
            
            
        #PONGO CORRECTAMENTE LA COLUMNA TIEMPO
        if type(file['Time'][0]) != str:
            file['Time'] = file['Time'].astype(str)
        for i in file['Time']:
            spliteo = i.split('.')
            con = spliteo[0]+'.'+spliteo[1]
            tsec.append(con)
            t_uni.append(spliteo[0])
        
        file['tiempo_dec(s)']=tsec
        file['tiempo_dec(s)'] = file['tiempo_dec(s)'].astype(float)
        file['t_uni']=t_uni
        file['t_uni'] = file['t_uni'].astype(float)
        
        #CALCULO N PAQUETES EN EL ULTIMO SEGUNDO
        q = deque()
        max_dif = 1
        #promedio del tiempo entre paquetes en las ventanas
        promedio_t_ultimo_s =[]
        promedio= 0
        n_paquetes = []
        ocurrencias = 0
        max_len_q = 0
        npaq_std = []
        
        for i in file['tiempo_dec(s)']:
            q.append(i)
            elimina_valores(q,dif_max =max_dif)
            max_len_q = max(max_len_q, len(q))
            promedio = max_dif/len(q)
            n_paquetes.append(len(q))
            promedio_t_ultimo_s.append(promedio)
       
            
        file['n_paquetes'] = n_paquetes
        file['n_paquetes'].astype(float)
        
        
#         print(file['n_paquetes'])
                #ESTADISTICAS CON LA FUNCION n paquetes
        n_paq_mean = []
        n_paq_std = []
        n_paq_min = []
        n_paq_max = []
        estadistica2 = file.n_paquetes.describe()
        for i in range(file.shape[0]):
            n_paq_mean.append(estadistica2[1])
            n_paq_std.append(estadistica2[2])
            n_paq_min.append(estadistica2[3])
            n_paq_max.append(estadistica2[7])
        file['n_paq_mean'] =  n_paq_mean
        file['n_paq_std'] =  n_paq_std
        file['n_paq_min'] =  n_paq_min
        file['n_paq_max'] =  n_paq_max
        
        dfs.append(file)
        #CREO COLUMNA PARA INDICAR SI N PAQUETES ES > 300 PAQUETES/s
#     df_con_indica_n_paquetes = []
#     for df in dfs:
#         alto_n_paquetes = [] 
#         for i in df.n_paquetes:
#             if i <300:
#                 alto_n_paquetes.append(0)
#             else:
#                 alto_n_paquetes.append(1)
#         df['alto_n_paquetes'] = alto_n_paquetes
#         df_con_indica_n_paquetes.append(df)
    
    dfs_ataque = []
    
    #CREO COLUMNA PARA INDICAR SI LOS DATOS VIENEN O NO DE UN ATAQUE
    for df in dfs:
        ataque = []
        no_ataque = []
        if lista_ataques:
            for i in range(df.shape[0]):
                ataque.append(1)
            df['ataque'] = ataque
        else:
            for i in range(df.shape[0]):
                no_ataque.append(0)
            df['ataque'] = no_ataque
        dfs_ataque.append(df)
    
    #CREO COLUMNA PARA INDICAR SI LOS DATOS VAN A UN PUERTO CALIENTE 21 O 22
    dfs_pcaliente = []
    for df in dfs_ataque:
        puerto_caliente = []    
        for fila in range(df.shape[0]):
            if type(df['Info'][fila]) != str:
                df['Info'] = df['Info'].astype(str)
            if (' 21 ' in df.Info[fila]) or (' 22 ' in df.Info[fila]):
    #             print(file.Info[fila],'--tenemos 21 o 22')
                puerto_caliente.append(1)
            else:
                puerto_caliente.append(0)
        df['puerto_caliente'] = puerto_caliente
        df['puerto_caliente'] = df['puerto_caliente'].astype(str)
        dfs_pcaliente.append(df)
        #CREO COLUMNA INDICANDO PUERTO DESTINO
    df_total1 = columna_puerto_destino(dfs_pcaliente)
    
    df_total = escaner_puertos(df_total1)
    
    #CREO COLUMNA PARA CONTAR POR CADA IP SI HAY IPS DESTINO QUE 
    df_total_concatenados = pd.concat(df_total)
#     sns.lineplot(y=n_paquetes, x=[i for i in range(len(n_paquetes))])
#     plt.xlabel('ocurrencias en el ultimo segundo')
#     plt.ylabel('peticiones/s')
#     df_total[['tiempo_dec(s)','n_paquetes','ataque']]
    return df_total_concatenados[['escaner','len_mean','len_std', 'len_min', 'len_max','puerto_destino','Protocol','Source','Destination','n_paq_mean','n_paq_std','n_paq_min','n_paq_max','n_paquetes','Length','puerto_caliente','ataque']]


def elimina_valores(cola, dif_max = 1):
    while(True):
        if cola[-1]-cola[0]>dif_max:
            cola.popleft()
        else:
#             cola.std()
            break
def calcula_origen_y_destino(url, lista_protocolos, lista_signo_fuente):
    df = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
    pos = 0
    lista_pares = []
    for protocolo in lista_protocolos:
        df = df[(df.Protocol == protocolo) & (df.Info.str.contains(lista_signo_fuente[pos]))]
        pos+=1
    for indice,fila in df.iterrows():
        lista_pares.append((fila.Source,fila.Destination))
    lista_pares = list(set(lista_pares))
    return lista_pares

def modelo_rf_cross_val(X,c = 5):
    print(X.columns)
#     print(X['ataque'])
#     X_train, X_test, y_train, y_test = train_test_split(X[['Length']], X['ataque'], test_size=0.3, random_state=42)

    y = X['ataque']
    X = X[GLOBAL_TRAIN_COLUMNS]
    
    numerical = X.select_dtypes(exclude=["category",'object']).columns
    categorical = X.select_dtypes(include=["object"]).columns
    t = [('cat', OneHotEncoder(), categorical), ('num', StandardScaler(), numerical)]
    transformer = ColumnTransformer(transformers=t)
    X_trans = transformer.fit_transform(X)
    scores_lista = []
    for i in range(2,5):
        rf = RandomForestClassifier(max_depth=i, random_state=0)
#         rf.fit(X_trans, y_train)
#         fitted_labels = rf.predict(X_test_transformed)
        scores = cross_val_score(rf,X_trans, y, cv=c)
        scores_lista.append(scores)
#         score = accuracy_score( fitted_labels,y_test)
        print('max_depth = ',i,' scores:', scores)
#        plt.scatter([i in range(1,10)], scores_lista)
#        plt.show()
    
    
#     return scores_lista

def modelo_rf(X):
#     print(X.columns)
    x = GLOBAL_TRAIN_COLUMNS
    X_train, X_test, y_train, y_test = train_test_split(X[x], X['ataque'], test_size=0.3, random_state=42)
    
    numerical = X_train.select_dtypes(exclude=["category",'object']).columns
    categorical = X_train.select_dtypes(include=["object"]).columns
    t = [('cat', OneHotEncoder(), categorical), ('num', StandardScaler(), numerical)]
    transformer = ColumnTransformer(transformers=t)
    X_train_transformed = transformer.fit_transform(X_train)
#     print(X_test)
    X_test_transformed = transformer.transform(X_test)
    mse = []
    for i in range(1,8):
        rf = RandomForestClassifier(max_depth=i, random_state=0)
        rf.fit(X_train_transformed, y_train)
        fitted_labels = rf.predict(X_test_transformed)
        score = accuracy_score( fitted_labels,y_test)
        print('max_depth = ',i,' score:', score)
        
        mse_ = mean_squared_error( y_test,fitted_labels)
        mse.append(mse_)
    print('longitud:',len(mse))
#     plt.scatter(fitted_labels, y_test)
    plt.scatter([range(1,8)], mse)
    
    
    return score

def modelo_lr(X):
#     X_train, X_test, y_train, y_test = train_test_split(X[['puerto_destino','alto_n_paquetes','Length','puerto_caliente']], X[['ataque']], test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X[GLOBAL_TRAIN_COLUMNS], X['ataque'], test_size=0.3, random_state=42)
    numerical = X_train.select_dtypes(exclude=["category",'object']).columns
    print('numerical',numerical)
    categorical = X_train.select_dtypes(include=["object"]).columns
    print('categorical',categorical)
    t = [('cat', OneHotEncoder(), categorical), ('num', StandardScaler(), numerical)]
    print(X_train.columns,'---',X_test.columns)
    transformer = ColumnTransformer(transformers=t)
    X_train_transformed = transformer.fit_transform(X_train)
    print(X_test)
    X_test_transformed = transformer.transform(X_test)
    
    logistic_model = LogisticRegression()
    
    hyperparameters = dict(C = np.logspace(0,4,10), penalty = ['l1','l2'], random_state = [0])
    
    gs = GridSearchCV(estimator = logistic_model, param_grid = hyperparameters, cv = 5)
    best_model = gs.fit(X_train_transformed,y_train)
    best_predictions = best_model.predict(X_test_transformed)
    score = accuracy_score(best_predictions,y_test)
    plt.scatter(y_test, best_predictions)
    
    return score



def elimina_valores(cola, dif_max = 1):
    while(True):
        if cola[-1]-cola[0]>dif_max:
            cola.popleft()
        else:
            break

#funcion que hace el promedio del tiempo entre paquetes en la última ventana temporal de 
#1s
#falta poner la condición de que sean paquetes entre las mismas IPs y de forma consecutiva
#está con la funcion def calcula_origen_y_destino
#funcion que hace el promedio del tiempo entre paquetes en la última ventana temporal de 
#1s
#falta poner la condición de que sean paquetes entre las mismas IPs y de forma consecutiva
#está con la funcion def calcula_origen_y_destino
def detecta_ventana(url, encoding='latin1', origen ='10.0.2.4', destino= '10.0.2.6'):
    print(url)
    file = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
#     file =file.iloc[900:,].reset_index()
#     file = file[(file.Source == origen)&(file.Destination == destino)].reset_index()
    tsec = []
    con = ''
    for i in file['Time']:
        spliteo = i.split('.')
    #     print(spliteo)
        con = spliteo[0]+'.'+spliteo[1]
    #     print(con)
        tsec.append(con)
        
    file['tiempo_dec(s)']=tsec
#     print('tsec')
    print(file['tiempo_dec(s)'])
    file['tiempo_dec(s)'] = file['tiempo_dec(s)'].astype(float)
    q = deque()
    max_dif = 1
    #promedio del tiempo entre paquetes en las ventanas
    promedio_t_ultimo_s =[]
    promedio= 0
    n_paquetes = []
    ocurrencias = 0
    max_len_q = 0
    for i in file['tiempo_dec(s)']:
        q.append(i)
#         print('q:',q)
        elimina_valores(q,dif_max =max_dif)
#        print(len(q))
        max_len_q = max(max_len_q, len(q))
        promedio = max_dif/len(q)
        n_paquetes.append(len(q))
#        print('paquetes: ',n_paquetes)
        
        promedio_t_ultimo_s.append(promedio)
#        print(promedio)
        if promedio < max_dif/1000:
#             print('promedio: ',promedio)
            ocurrencias+=1
            if ocurrencias ==1:
                print('AVISO. Posible ataque')
            elif ocurrencias > 1:
                print('¡¡¡Esto es una ataque!!!')
#                 break
    print('Max len q:', max_len_q)
    print('Max n paquetes:', max(n_paquetes))
    sns.set(style="darkgrid")

    # Plot the responses for different events and regions
    df=pd.DataFrame()
    df['peticiones_ultimo_segundo'] = n_paquetes
    df['linea_temporal'] = [i for i in range(len(n_paquetes))]
#    sns.lineplot(x='peticiones_ultimo_segundo', y='linea_temporal', data=df)
    sns.lineplot(y=n_paquetes, x=[i for i in range(len(n_paquetes))])
    plt.xlabel('ocurrencias en el ultimo segundo')
    plt.ylabel('peticiones/s')
    
#     return file

def calcula_origen_y_destino(url, lista_protocolos, lista_signo_fuente):
    df = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
    pos = 0
    lista_pares = []
    for protocolo in lista_protocolos:
        df = df[(df.Protocol == protocolo) & (df.Info.str.contains(lista_signo_fuente[pos]))]
        pos+=1
    for indice,fila in df.iterrows():
        lista_pares.append((fila.Source,fila.Destination))
    lista_pares = list(set(lista_pares))
    return lista_pares

def open_file(url, encoding='latin1'):
    file = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
    return file

def open_file2(list_url, encoding='latin1'):
    for i in list_url:
        print(str(i))
        file = pd.read_csv(i,encoding='latin1',sep = ';',error_bad_lines=False)
        file.head()
        titulo = i[:-4] +'after'+'.csv'
        print(titulo)
        file.to_csv(titulo, sep = ';')
        print('------------ % null values ------------')
        print(file.isna().sum()*100/len(file))
        print('-----------------------------')
    return file


def trafico_paquetes(url, encoding='latin1'):
    
    file = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
    file['tiempo'] = file.Time.apply(lambda x: float(x.split('.')[0]))
    peticiones_por_seg = []
    segundo = []
    for name, group in file.groupby('tiempo'):
    #     print(group[(group['Source'] == '10.0.2.4') | (group['Source'] ==  '10.0.2.6')])
        segundo.append(int(name))
        peticiones_por_seg.append((group[(group['Source'] == '10.0.2.4') | (group['Source'] ==  '10.0.2.6')]).shape[0])
    df=pd.DataFrame()
    df['tiempo'] = segundo
    df['tiempo']=df['tiempo'].apply(lambda x: float(x))
    df['peticiones_por_s']= peticiones_por_seg
    sns.set(style="darkgrid")

    # Plot the responses for different events and regions
    sns.lineplot(x="tiempo", y="peticiones_por_s",
                 data=df)
    plt.xlabel('tiempo(s)')
    plt.ylabel('peticiones/s')
    return file

def df_trafico(url, encoding='latin1'):
    file = pd.read_csv(url,encoding='latin1',sep = ';',error_bad_lines=False)
    file['tiempo'] = file.Time.apply(lambda x: float(x.split('.')[0]))
    peticiones_por_seg = []
    segundo = []
    for name, group in file.groupby('tiempo'):
    #     print(group[(group['Source'] == '10.0.2.4') | (group['Source'] ==  '10.0.2.6')])
        segundo.append(int(name))
        peticiones_por_seg.append((group[(group['Source'] == '10.0.2.4') | (group['Source'] ==  '10.0.2.6')]).shape[0])
    df=pd.DataFrame()
    df['tiempo'] = segundo
    df['tiempo']=df['tiempo'].apply(lambda x: float(x))
    df['peticiones_por_s']= peticiones_por_seg
    return df