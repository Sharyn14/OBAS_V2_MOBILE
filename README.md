# OBAS V2 MOBILE
### Detección y segmentación de áreas quemadas mediante inteligencia artificial

**Proyecto académico de Ingeniería de Sistemas**

**Integrantes:**
- Sharyn Galviz
- Santiago Falla
- Nicole Peña

## 1. Descripción del proyecto

OBAS V2 Mobile es una solución tecnológica que utiliza inteligencia artificial para identificar y segmentar áreas quemadas a partir de imágenes satelitales.

El proyecto integra un modelo de aprendizaje profundo con una aplicación móvil desarrollada en Flutter para Android.

Su finalidad es apoyar la revisión preliminar de zonas afectadas por incendios y aportar información útil para el seguimiento ambiental.

## 2. Problema identificado

Después de un incendio, identificar las áreas afectadas mediante imágenes satelitales puede requerir tiempo y conocimientos especializados.

La revisión manual de grandes cantidades de información puede retrasar el análisis inicial de las zonas quemadas.

OBAS V2 Mobile propone una herramienta de apoyo que permite obtener una segmentación automática para facilitar esta revisión.

## 3. Nuestra solución

El sistema utiliza un modelo de segmentación semántica binaria que clasifica los píxeles en dos categorías:

- **Área no quemada:** regiones que el modelo identifica como no afectadas.
- **Área quemada:** regiones que el modelo identifica como afectadas por el incendio.

El modelo fue preparado para su utilización en dispositivos móviles mediante TensorFlow Lite.

La aplicación permite realizar inferencia local, sin necesidad de conexión a internet durante el análisis.

## 4. Tecnologías utilizadas

- Python
- TensorFlow y Keras
- TensorFlow Lite
- Arquitecturas U-Net, ResNet50 y MobileNetV2
- Knowledge Distillation
- Flutter y Dart
- Android
- Kaggle
- GitHub

## 5. Arquitectura del modelo

El proyecto incluye un modelo Teacher basado en ResNet50 y un modelo Student basado en MobileNetV2.

Se trabajó con técnicas de destilación de conocimiento para estudiar la transferencia de información entre modelos.

Para el despliegue móvil se dispone de un modelo TensorFlow Lite con optimización Float16.

Los notebooks del repositorio documentan el entrenamiento, las evaluaciones, la conversión y los análisis realizados.

## 6. Resultados obtenidos

En la auditoría de 50 parches se obtuvieron los siguientes resultados:

| Métrica | Resultado |
|---|---|
| F1-score | 0,8089 |
| Precisión | 0,9217 |
| Recall | 0,7207 |
| IoU | 0,6792 |
| Tamaño del modelo TFLite Float16 | 15,31 MB |
| Latencia promedio medida en Kaggle | 253,31 ms |

**Aclaración:** la latencia de 253,31 ms fue medida en Kaggle y no corresponde al tiempo de ejecución en un teléfono.

Los resultados de auditoría deben interpretarse considerando las limitaciones del conjunto de evaluación y la necesidad de validación adicional.

## 7. Instalación de la aplicación

La aplicación se distribuye como un archivo APK para Android.

**Pasos de instalación:**

1. Descargar el archivo `app-release.apk` desde el enlace de descarga del proyecto.
2. Abrir el archivo APK en un dispositivo Android.
3. Si Android solicita autorización para instalar aplicaciones desde esa fuente, permitirla siguiendo las indicaciones del sistema.
4. Confirmar la instalación.
5. Abrir OBAS V2 Mobile.

La aplicación funciona localmente después de instalarse y no requiere conexión a internet para ejecutar el análisis de archivos compatibles.

**Descarga del APK:**
Para instalar:

Descargar el APK desde el enlace anterior.

Abrir el archivo descargado en un teléfono Android.

Autorizar la instalación si Android lo solicita.

Instalar y abrir OBAS V2 Mobile.

La aplicación permite realizar análisis locales sin conexión a internet utilizando archivos satelitales compatibles .npy o .bin.

Versión: 1.0.0

Tamaño del APK: aproximadamente 80,6 MB.
https://github.com/Sharyn14/OBAS_V2_MOBILE/releases/download/v1.0.0/app-release.3.apk


## 8. Uso de la aplicación

1. Abrir OBAS V2 Mobile.
2. Seleccionar la opción de carga de archivos.
3. Elegir un archivo satelital compatible en formato `.npy` o `.bin`.
4. Ejecutar el análisis.
5. Visualizar el resultado de segmentación.

La aplicación no está diseñada para procesar directamente fotografías JPG comunes. Utiliza datos satelitales preparados en los formatos compatibles.

### Demostración de funcionamiento sin internet

https://youtube.com/shorts/Lojq3crfCyY?si=ItbEur5-JUPWVJ9Q

## 9. Cómo ejecutar los notebooks

Los notebooks están disponibles en la carpeta `notebooks/`.

Para revisarlos o ejecutarlos:

1. Ingresar a Kaggle Notebooks.
2. Importar el archivo `.ipynb` correspondiente.
3. Configurar el entorno con las dependencias utilizadas en el proyecto.
4. Vincular los datasets y archivos de modelos requeridos.
5. Ejecutar las celdas en el orden documentado.

**Importante:** la reproducción completa del entrenamiento requiere disponer de los datos, modelos y recursos computacionales correspondientes. El notebook histórico conserva código original y no representa por sí solo un nuevo entrenamiento ejecutado.

## 10. Datasets utilizados

Se utilizaron recursos de imágenes satelitales relacionados con incendios y áreas quemadas.

**Wildfires CEMS:**

https://huggingface.co/datasets/links-ads/wildfires-cems

**Dataset publicado en Mendeley Data:**

https://data.mendeley.com/datasets/5d8kxmmwtz/1

El sistema trabaja con dos clases de segmentación: área quemada y área no quemada.

Los datasets completos no se incluyen en el repositorio para evitar almacenar archivos de gran tamaño.

## 11. Estructura del repositorio

- `README.md`: documentación general del proyecto.
- `app/`: código fuente de la aplicación Flutter.
- `notebooks/`: notebooks de entrenamiento, evaluación y auditoría.
- `model/`: modelo TensorFlow Lite y archivo de etiquetas.
- `docs/`: informe de auditoría, presentación final y capturas.
- `codigo_historico/`: código original complementario, si se incluye.

## 12. Auditoría y validación

El proyecto cuenta con una auditoría de 50 parches para evaluar el comportamiento del modelo.

Se analizaron métricas de precisión, recall, F1 e IoU.

También se realizaron pruebas de robustez ante alteraciones de las imágenes, incluyendo ruido, cambios de brillo y desenfoque.

Los resultados evidencian que el modelo puede cometer errores de segmentación, por lo que sus predicciones requieren revisión humana.

La evaluación presentada no debe interpretarse como una validación independiente definitiva para uso operativo.

## 13. Limitaciones

- El modelo puede producir falsos positivos y falsos negativos.
- Su rendimiento puede disminuir cuando las imágenes presentan alteraciones.
- La aplicación requiere archivos satelitales compatibles.
- Las métricas de Kaggle no equivalen necesariamente al rendimiento en todos los dispositivos Android.
- Los resultados son de apoyo y no sustituyen una evaluación técnica especializada.

## 14. Impacto y propuesta

OBAS V2 Mobile busca apoyar el monitoreo ambiental mediante una herramienta portátil para la revisión preliminar de áreas quemadas.

Como siguiente etapa se propone explorar un piloto supervisado con Cortolima, sujeto a su interés y autorización, utilizando imágenes satelitales reales y validación de especialistas.

El objetivo es evaluar la utilidad del sistema y establecer oportunidades de mejora antes de considerar una aplicación operativa.

## 15. Pitch final del proyecto

Presentación del proyecto OBAS V2 Mobile:

https://youtu.be/aVknSCkavds

## 16. Documentación y evidencias

La carpeta `docs/` contiene la documentación disponible del proyecto, incluyendo el informe de auditoría, la presentación del pitch y las capturas de funcionamiento.

La carpeta `notebooks/` contiene los cuadernos de trabajo y la auditoría de la Semana 15.

## 17. Equipo de desarrollo

**Sharyn Galviz**

**Santiago Falla**

**Nicole Peña**

Proyecto académico OBAS V2 Mobile.

---

**OBAS V2 Mobile — Inteligencia artificial al servicio del monitoreo ambiental.**
