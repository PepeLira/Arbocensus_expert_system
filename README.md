# Arbocensus Expert System
Arbocensus es un proyecto impulsado por la Facultad de Ingeniería de la Universidad de los Andes para la detección y censado del arbolado urbano. Busca ser una plataforma de uso publico que permite a personas corrientes realizar labores de censado, realizando campañas para obtener datos de manera eficiente e integrando técnicas de visión artificial para obtener los parámetros de cada árbol mediante un grupo de fotografías.

En concreto este repositorio esta destinado a la implementación del **Sistema Experto**. Tomando como base las imágenes recolectadas por la plataforma Arbocensus este sistema es capaz de extraer métricas importantes para el censado de arboles como los son la altura general, el diámetro a la altura del pecho (DAP), inclinación y alto de las ramas principales. Esta fue implementada en un paquete de Python disponible en `src/arbocensus_expert_system/`, diseñando un flujo que integra modelos como:
 - [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) para la extracción de mascaras de arboles.
 - [Depth Anything](https://github.com/LiheYoung/Depth-Anything) para filtrar y mejorar el area de búsqueda del árbol y las tarjetas de referencia.
 - [Grounding Dino](https://github.com/IDEA-Research/GroundingDINO) para la búsqueda de las tarjetas de referencia. 
 - [Clasificador Resnet18](https://github.com/PepeLira/Arbocensus_expert_system/blob/main/notebooks/species_clasification.ipynb) para la detección de las especies de los arboles. Fue entrenado con las imágenes disponibles en Arbocensus y la es especies catalogadas por [Arbotag](https://web.arbotag.cl/) 

**COMENTARIOS...** 
- Las tarjetas de referencia se encuentran a los pies de cada árbol, corresponden a una tarjeta tamaño estándar, como una tarjeta del banco o pase para el transporte publico. Es mediante a esta que es posible realizar una taza de pixeles a un sistema métrico, tomando como referencia sus dimensiones.
- Para entender mejor el flujo de trabajo antes de desarrollar el paquete, se realizo un Jupyter notebook probando los diferentes modelos. [Este se encuentra disponible en `./notebooks/exploration.ipynb`.](https://github.com/PepeLira/Arbocensus_expert_system/blob/main/notebooks/exploration.ipynb)
- El proceso de entrenamiento del clasificador esta disponible en [el siguiente jupyter notebook](https://github.com/PepeLira/Arbocensus_expert_system/blob/main/notebooks/species_clasification.ipynb).

## Guía de Inicio
### 1. Instalación de PyTorch
Antes de comenzar debemos preparar un par de cosas antes de instalar nuestra primera dependencia, `PyTorch`. En concreto este proyecto se beneficia considerablemente de poder trabajar con una tarjeta gráfica dedicada con sus drivers CUDA compatibles. Para esto debemos asegurarnos de contar con los drivers compatibles como se especifica en su [documentación] (https://pytorch.org/) (CUDA 12.4, 12.1 o 11.8 al momento de escribir este documento). Considerando que estos no suelen ser la ultima versión disponible de Nvidia, debemos elegir una [versión compatible](https://developer.nvidia.com/cuda-toolkit-archive).

Una vez instalados los drivers de nuestra tarjeta gráfica, comenzamos la instalación de `PyTorch` como indica su [documentación] (https://pytorch.org/). 

Verificamos que se realizo de manera correcta abriendo una consola de python (o ejecutando un archivo) con las siguientes lineas:
```
import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU is available: {gpu_name}")
    # Check the current device being used by torch
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("GPU is not available, running on CPU")
```
### 2. Dependencias del proyecto

Nos dirigimos al directorio del paquete:
```
$ cd .\src\
```

Instalamos las dependencias listadas en `requirements.txt`
```
pip install -r requirements.txt
```

### 3. Descargar pesos pre-entrenados 

Como se menciono anteriormente este proyecto integra multiples modelos. Lamentablemente los pesos pre-entrenados suelen ser de un tamaño mayor al permitido en Github, por lo que el estándar sugiere descargarlos de diferentes fuentes. En concreto este proyecto depende dos pesos:

- `sam_vit_h_4b8939.pth` disponible en el [repositorio oficial de SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file) o en [este link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
- `species_resnet18.pth` es un respaldo de nuestro modelo de clasificación preliminar, disponible en este link.

Una vez descargados debemos crear una carpeta en el proyecto con el nombre `./weights_and_checkpoints/` (y asegurarnos de no subirla por accidente en un commit.)

### 4. Preparar nuestras variables de entorno.
Si observamos dentro del directorio `src/arbocensus_expert_system/` encontraremos un archivo `parameters.env`. Debemos crear una copia de este y renombrarla a `my_parameters.env` y de esta manera no correr el riesgo de publicar información sensible por error. 

Algunos de los campos más importantes a especificar son:
- SPECIES_CLASSIFIER: la ruta donde se encuentran los pesos pre-entrenados de nuestro clasificador de especies `species_resnet18.pth`.
- SAM_CHECKPOINT: la ruta donde se encuentran los pesos de SAM.
- TREE_IMAGE_DESTINY: donde queremos almacenar las imagenes obtenidas mediante la API de Arbocensus.
- SECRET_KEY: la API key para poder interactuar con la API de Arbocensus.
- TEST_TREE_IMAGES_PATH, TEST_RESULTS_PATH: las rutas donde podremos dejar imagenes de prueba con las que trabajar.
- MARKS_JSON_PATH: la ruta a un archivo de marcas (coordenadas donde se encuentra cada tarjeta en cada imagen de árbol).

### 5. Probar la aplicación localmente
Listo si queremos comenzar a realizar detecciones de arboles podremos llamar a algunos paquetes de la aplicación, por ejemplo `tree_reviewer` se encarga de evaluar una imagen e imprimir los resultados, `expert_system` nos permite analizar todas las imágenes de TEST_TREE_IMAGES_PATH. 

Podemos ejecutarlos llamando al paquete:
```
$cd ./src/
```
```
python -m arbocensus_expert_system.expert_system
```

obtendremos los resultados en TEST_RESULTS_PATH.

Para facilitar un poco este proceso podemos realizar pruebas en el jupyter notebook `sand_box.ipynb`.

### 6. Interfaz con la API 
Ejecutando el paquete `arbocensus_api_interface` podremos realizar llamadas a la API de Arbocensus de manera automática. Sin embargo este paso require una dependencia extra...

**Comandos de Google Cloud para descargar imágenes:** Esto se hace mediante [gsutil](https://cloud.google.com/storage/docs/gsutil?hl=es-419) en donde por detrás la API de Arbocensus nos genera un link para descargar las imágenes desde google cloud. Para esto se debe instalar gsutil en el PATH de nuestro sistema siguiendo [esta guía](https://cloud.google.com/storage/docs/gsutil_install?hl=es-419) (Se deberá iniciar sesión con una cuenta de google autorizada, consultar con encargados).

Luego para comenzar a procesar las imágenes pendientes en la cola de una campaña podremos ejecutar:
```
$cd ./src/
```
```
python -m arbocensus_expert_system.arbocensus_api_interface
```

**Comentario:** En caso de necesitar programar la ejecución cíclica de esta tarea, podremos llamar a este script mediante un cron dependiendo del sistema operativo con el que se este trabajando.

## Pendientes

1. Agregar una estrategia para obtener la inclinación de los troncos de los arboles con respecto al suelo.  
2. Continuar el desarrollo del modelo de clasificación de especies.  

