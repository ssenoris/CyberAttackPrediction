# Predicción de ciberataques


![alt text](im.jpg)

Se han analizado diferentes tipos de ataques cibernéticos junto con tráfico normal y se han etiquetado estos datos para apicar un modelo de Machine Learning que aprenda a detectar tráfico malicioso.

Las características más destacables que se han detectado son la frecuencia del envío de paquetes y el tamaño que tienen, aunque existen otras características propias de algunos ataques como es el escaneo de puertos o las repetidas llamadas a puertos de control en remoto, por ejemplo. Por estos motivos los datos en crudo se han tratado para calcular y enfatizar estas catacterísticas que puedan hacer que el algoritmo aprenda mejor.

Inicialmente el dataset contiene las siguientes columnas:

* Time: tiempo, lo marca la herramienta que captura el tráfico y se mide en segundos. La diferencia en tiempos entre los paquetes indica la frecuencia a la que se producen los envíos.
* Source: es la IP origen.
* Destination: es la IP destino.
* Protocol: protocolo de transmisión de información.
* Length: numero de caracteres del paquete.
* Info: contiene información de los puertos origen y destino y del tipo de petición que se hace, entre otros.

Se realizaron distintos cálculos en el dataset como el número de paquetes por segundo y las diferentes variables estadísticas de esta columna (máximo, mínimo, media y desviación típica), así como las mismas variables estadísticas para el tamaño de paquete.

Los datos han sido tratados de la siguiente manera:

El cálculo del número de paquetes por segundo enviados se hizo de manera que en cada instante de tiempo en el que la herramienta nos proporciona datos, tenemos el cálculo de los paquetes/s del último segundo. De la columna info se ha extraído el dato de puerto destino y se ha puesto en una columna aparte. También se ha etiquetado si existe o no escaner de puertos (desde una misma IP realizar peticiones a muchos puertos distintos puede ser una estrategia previa a un ataque) y también se ha etiquetado la presencia de puertos 21 y 22 por ser puertos que solicitan control de máquinas en local y en remoto, respectivamente.

Los datos han sido debidamente transformados a código binario, aplicando distintas transformaciones en función de la naturaleza del mismo (numérico o categórico). Para el modelo de ML se ha empleado un RanomForestClassifier. El modelo hace una predicción de si el tráfico es o no es un ataque en base a un etiquetado previo (Supervised Learning). Los hiperparámetros del modelo han sido optimizados con un RandomizedSearchCV. El modelo se ha validado con un cross_validation y con LearningCurve.