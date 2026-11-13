OBAS V2 — RECURSOS VISUALES COMPLEMENTARIOS

Este ZIP complementa el paquete del modelo OBAS V2.

CONTENIDO
- tolima/imagenes/: imágenes y capas raster del caso Tolima disponibles.
- tolima/resultados/: máscara predicha, fichas y resúmenes geográficos disponibles.
- gradcam/: visualización Grad-CAM de un experimento previo y sus métricas, si están disponibles.
- documentacion/: inventario de los recursos incluidos.

USO EN FLUTTER
1. Integrar los recursos de Tolima en una pestaña de resultados de un experimento previo.
2. Mostrar la visualización Grad-CAM en una pestaña de interpretabilidad.
3. Mantener separados estos resultados históricos de las predicciones nuevas.

IMPORTANTE
- La Grad-CAM incluida corresponde a un experimento previo, no necesariamente
  al modelo Float16 utilizado en la evaluación de los 20 parches.
- Los GeoTIFF conservan sus datos raster y metadatos geográficos; no todos los
  widgets de imagen de Flutter pueden mostrarlos directamente. Para visualización
  móvil conviene generar PNG de previsualización y conservar los GeoTIFF como datos.
- Las hectáreas y la biomasa deben mostrarse con sus supuestos y limitaciones.
- Este paquete no contiene el código de la aplicación Flutter.
