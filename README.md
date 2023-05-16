# Proyecto Individual Henry Matias Cejas
<!DOCTYPE html>
<html>
  <head>
    <img src="https://stonkstutors.com/wp-content/uploads/2022/07/Soy-Henry-Entiende-como-funciona-la-plataforma-y-si-vale-la-pena.jpg" alt="henry" width="900" height="300">
  </head>
  <body>
    <div style="display: flex; align-items: center;">
    <h1><br><br>Proyecto Machine Learning Operations (MLOps)</h1>
    <img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png" alt="MLOps" width="120" height="120">
    </div>
    <h2>Proyecto: </h2>
    <p>El proyecto consiste en trabajar como Data Scientist en una start-up que provee servicios de agregación de plataformas de streaming.
    <br>Para ello nos entregan un archivo .CSV en crudo con los datos de peliculas que emiten en la plataforma. <br>Requisitos:
    <ul><li>Realizar ciertas consultas a los datos</li>
    <li>Realizar un modelo de recomendacion de peliculas que al ingresar un titulo nos devuelva una recomendacion de 5 peliculas similares</li>
    <li>Realizar una API lista para su consumo con los puntos anteriores</li>
    </ul> </p>
    <h2>Consulta de Datos</h2>
    <p><ul>
    <li><strong>peliculas_mes:</strong> Se ingresa el mes y la funcion retorna la cantidad de peliculas que se estrenaron ese mes (nombre del mes, en str, ej: Enero) historicamente</li>
    <li><strong>peliculas_dia:</strong> Se ingresa el dia y la funcion retorna la cantidad de peliculas que se estrenaron ese dia (de la semana, en str, ejemplo 'Lunes') historicamente</li>
    <li><strong>franquicia:</strong> Se ingresa la franquicia, retornando la cantidad de peliculas, ganancia total y promedio</li>
    <li><strong>peliculas_pais:</strong> Se ingresas el pais, retornando la cantidad de peliculas producidas en el mismo</li>
    <li><strong>productoras:</strong> Se ingresas la productora, retornando la ganancia total y la cantidad de peliculas que produjeron</li>
    <li><strong>retorno:</strong> Se ingresas la pelicula, retornando la inversion, la ganancia, el retorno y el año en el que se lanzo</li>
    </ul></p>
    <h2>Modelo de recomendación de películas </h2>
    <p>Este proyecto realiza un modelo de recomendación de películas basado en contenido. Este tipo de modelo utiliza información sobre las películas, como su título, género, actores y directores, para encontrar películas similares.<br>Una vez que se tiene la fuente de datos, se realizan los siguientes pasos:.</p>
    <ul>
    <li><strong>Preprocesamiento de datos:</strong> Se limpian los datos y transforman en un formato adecuado para el modelo.</li>
    <li><strong>Creación de una matriz de características:</strong> Para cada película en la fuente de datos, se crea un vector de características que describa la película.</li>
    <li><strong>Cálculo de similitudes:</strong> Para encontrar películas similares, se calcula la similitud entre los vectores de características de todas las películas en la fuente de datos. Para ello se utiliza la similitud del coseno.</li>
    <li><strong>Recomendación de películas:</strong> Se encuentran las 5 películas más similares al título recibido como parámetro y se lo muestra en una lista</li>
    </ul></p>
    <h2>Implementacion/Uso</h2> 
    <p><ul>
    <li>En el archivo ETL.ipynb se encuentra todo lo referido a el proceso de transformacion de los datos crudos que nos proporcionan. Para mas info remitirse a dicho archivo </li>
    <li>En el archivo EDA_ML.ipynb se encuentra el Analisis Exploratorio de Datos(EDA). Para mas info remitirse a dicho archivo </li>
    <li>En el archivo main.py se encuentra el achivo principal, el cual se deve ejecutar en la API</li>
    </ul></p>
    <h2>Bibliografía</h2> 
    <p>Consignas del Proyecto >> <a href="https://github.com/HX-PRomero/PI_ML_OPS/blob/main/Readme.md">Enlace al archivo</a></p>       
         
  </body>
</html>

