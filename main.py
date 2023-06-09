###### Importo librerias y cargo el dataframe con los datos para las consultas ######
from fastapi import FastAPI
import pandas as pd
import numpy as np
import locale
import ast
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse ,JSONResponse
df = pd.read_csv("DataSets/Peliculas.csv")

app = FastAPI()

###### Genero los @app para la api ######
# http://127.0.0.1:8000/
###### .get("/") >> ventana inicial de la api. def index >> funcion inicial de la api donde carga y  muestra el readme ######
@app.get("/",response_class =HTMLResponse )
def index():
    readme_path = "README.html"  
    return FileResponse(readme_path)


###### .get('/peliculas_mes/ >> Ventana donde se consulta la funcion pelicula_mes ######
###### peliculas_mes(mes) >> Realizo un groupby de la columna month_name y luego selecciono el indicado por el parametro ######
@app.get('/peliculas_mes/{mes:str}')
def peliculas_mes(mes):
    '''Funcion donde se ingresa el mes y retorna la cantidad de peliculas que se estrenaron ese mes historicamente'''
    '''Al ingresar el mes recuerde cargarlo con mayuscula ejemplo: Enero, Febrero, Marzo, etc. '''
    aux = df.groupby("month_name").size()
    respuesta = aux [mes]
    return {'mes':mes, 'cantidad':str(respuesta)}


###### .get('/peliculas_dia/ >> Ventana donde se consulta la funcion peliculas_dia ######
###### peliculas_dia(dia) >> Realizo un groupby de la columna day_name y luego selecciono el indicado por el parametro ######
@app.get('/peliculas_dia/{dia:str}')
def peliculas_dia(dia):
    '''Funcion donde se ingresa el dia y retorna la cantidad de peliculas que se estrenaron ese dia historicamente'''
    aux = df.groupby("day_name").size()
    respuesta = aux [dia]
    return {'dia':dia, 'cantidad':str(respuesta)}


###### .get('/franquicia/ >> Ventana donde se consulta la funcion franquicia ######
###### franquicia >> Realizo un query donde obtengo la cantidad, el total y el promedio de la columna revenue(ganancia) ######
@app.get('/franquicia/{franquicia:str}')
def franquicia(franquicia):
    '''Funcion donde se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio de la franquicia'''
    cantidad = df.query("belongs_to_collection == @franquicia")['revenue'].count()
    ganancia = df.query("belongs_to_collection == @franquicia")['revenue'].sum()
    promedio = df.query("belongs_to_collection == @franquicia")['revenue'].mean()
    return {'franquicia':franquicia, 'cantidad de peliculas':str(cantidad), 'ganancia_total':str(ganancia), 'ganancia_promedio':str(promedio)}


###### .get('/peliculas_pais/ >> Ventana donde se consulta la funcion peliculas_pais ######
###### peliculas_dia(dia) >> Creo un nuevo dataframe donde filtro el parametro y luego cuento la cantidad de registros  ######
@app.get('/peliculas_pais/{pais:str}')
def peliculas_pais(pais):
    '''Funcion donde se ingresas el pais, retornando la cantidad de peliculas producidas en el mismos''' 
    df_filtrado = df[df['production_countries'].apply(lambda x: pais in str(x))]
    cantidad_peliculas = df_filtrado["production_countries"].count()
    return {'pais':pais, 'cantidad':str(cantidad_peliculas)}


###### .get('/productoras/ >> Ventana donde se consulta la funcion productoras ######
###### productoras >> Creo un nuevo dataframe donde filtro el parametro y luego cuento la cantidad de registros y sumo "reveneu"(ganancias) para obtener el total  ######
@app.get('/productoras/{productora:str}')
def productoras(productora):
    '''Funcion donde se ingresas la productora, retornando la ganancia total y la cantidad de peliculas que produjeron'''
    df_filtrado = df[df['production_companies'].apply(lambda x: productora in str(x))]
    cantidad = df_filtrado["production_companies"].count()
    ganancia= df_filtrado["revenue"].sum()
    return {'productora':productora, 'ganancia_total':str(ganancia), 'cantidad':str(cantidad)}


###### .get('/retorno/ >> Ventana donde se consulta la funcion retorno ######
###### retorno >> Creo un nuevo dataframe donde filtro el parametro y luego cargoy muestro la inversion, la ganancia, el retorno y el año de lanzamiento de la pelicula ######
@app.get('/retorno/{pelicula:str}')
def retorno(pelicula):
    '''Funcion donde se ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo '''
    df_filtrado = df[df['title'].apply(lambda x: pelicula.strip() == str(x.strip()))]
    inversion= df_filtrado["budget"].sum()
    ganancia= df_filtrado["revenue"].sum()
    retornoo= df_filtrado["return"].sum()
    anio = pd.to_datetime(df_filtrado["release_date"]).dt.year
    return {'Pelicula':  pelicula,'Inversion':str(inversion),'Ganacia':str(ganancia),'Retorno':str(retornoo),'Año':str(anio.item())}



##################################### Modelo Machine Learning  ##############################################################
###### Importo librerias y cargo el archivo luego de aplicar el ETL y el EDA ######
from sklearn.metrics.pairwise import cosine_similarity
data = pd.read_csv('DataSets/PeliculasML.csv', low_memory=False)  

###### Recorto el Dataframe para poder aplicar el modelo  ######
n=6000 
datos = data.head(n)

###### Alimento la variable X con los nombre de las columnas para poder aplicar el modelo ######
X = datos[['belongs_to_collection', 'genres', 'original_language', "popularity", "production_companies", "release_date", "runtime", "status", "vote_average", "return"]] 

###### Aplico el modelo de Machine Learning de similaridad del coseno ###### 
matris = cosine_similarity(X)

###### recomendacion >> Funcion en donde obtengo el indice del titulo segun el parametro luego obtengo los indices de peliculas similares ######
###### Finalmente busco en el dataframe el titulo de las peliculas similares y las retorno para visualizarlas en una lista ######
@app.get('/recomendacion/{titulo:str}')
def get_recomendacion(titulo):
    '''Funcion de Machine Learning donde se carga un titulo de una pelicula y el algoritmo te recomienda 5 titulos similares '''
    top_n=5
    indice_titulo = data[data['title'] == titulo].index[0]  
    resultado_matris = matris[indice_titulo]
    indices = resultado_matris.argsort()[-top_n-1:-1][::-1]  
    recomendacion = data.loc[indices,"title"]  
    recomendacion = recomendacion.values.tolist()
    return {'lista recomendada': str(recomendacion)}


