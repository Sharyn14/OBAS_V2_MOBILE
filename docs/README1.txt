OBAS V2 — PAQUETE DE DEMOSTRACIÓN FLUTTER OFFLINE

CONTENIDO
- modelos/: modelos TensorFlow Lite Float16 y, si está disponible, Float32.
- datos/parches/: 20 entradas originales en formato NPY, CHW (3,128,128).
- datos/mascaras_reales/: máscaras de referencia.
- datos/previsualizaciones/: imágenes PNG originales disponibles en el manifiesto.
- resultados/mascaras_predichas/: predicciones binarias NPY.
- resultados/fichas/: fichas comparativas por parche.
- resultados/metricas/: métricas, manifiesto corregido e inventario.
- interpretabilidad/: visualización Grad-CAM de un experimento anterior, si está disponible.

MODELO RECOMENDADO PARA LA DEMOSTRACIÓN MÓVIL
student_baseline_float16.tflite

ENTRADA DEL MODELO
- Forma esperada: [1,128,128,3]
- Tipo: float32
- Las imágenes NPY están almacenadas en formato CHW; para inferencia se convierten a HWC.
- No renormalizar automáticamente los valores almacenados.

SALIDA
- Forma esperada: [1,128,128,1]
- Aplicar umbral 0.5 para obtener una máscara binaria, según la evaluación incluida.

RESULTADOS DE LA MUESTRA
- 20 parches del split test.
- Dice global agregado e IoU global agregado disponibles en resumen_global_20_parches.json.
- La inferencia mediana observada en Kaggle fue de aproximadamente 31 ms por parche.
- Estos resultados corresponden a la muestra evaluada; no son una garantía de rendimiento en campo.

LIMITACIONES
- El parche OBAS_zone1 / patch_id 2179 reemplaza el caso original número 1.
- La representación de un parche CEMS contiene valores mayores que 1; se conservó sin renormalización.
  La compatibilidad de esa distribución con el entrenamiento debe verificarse.
- La imagen Grad-CAM incluida procede de un experimento previo y no debe interpretarse como Grad-CAM
  calculada por el modelo Float16 evaluado en esta sesión.
- Las hectáreas requieren resolución espacial conocida. La biomasa no se considera una medición
  real sin datos de campo y un modelo ecológico validado.
- Para uso en producción, validar la inferencia con imágenes nuevas, metadatos espaciales y pruebas
  en el dispositivo Android objetivo.

Este paquete es una base de demostración offline; no reemplaza la validación operacional.
