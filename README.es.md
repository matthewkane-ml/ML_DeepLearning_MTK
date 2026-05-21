# Clasificador de Imágenes con CNN — Perros vs. Gatos

> Una red neuronal convolucional inspirada en VGG16 que clasifica imágenes como perros o gatos, construida desde cero con TensorFlow/Keras y entrenada con parada anticipada y guardado del mejor modelo.

---

## Problema

La clasificación de imágenes es una de las tareas más fundamentales en visión por computadora. Este proyecto implementa una red neuronal convolucional profunda modelada sobre la arquitectura VGG16 para distinguir entre imágenes de perros y gatos — demostrando el pipeline completo desde la carga y preprocesamiento de imágenes hasta el entrenamiento, evaluación e inferencia del modelo.

## Conjunto de datos

- **Fuente:** Dataset local de imágenes organizado en subcarpetas `dog/` y `cat/` dentro de `data/raw/`
- **Formato:** Imágenes RGB, redimensionadas a 224 × 224 píxeles para entrada compatible con VGG16
- **División:** 80% entrenamiento / 20% validación mediante `ImageDataGenerator(validation_split=0.2)`
- **Clases:** `dog` (índice 0), `cat` (índice 1)

## Metodología

1. **Carga de datos:** `ImageDataGenerator` de Keras con normalización `rescale=1./255` y división 80/20 entrenamiento/validación. Imágenes cargadas en lotes de 8 con resolución 224×224.
2. **Arquitectura:** CNN estilo VGG16 construida desde cero con `Sequential` de Keras:
   - 5 bloques convolucionales con profundidad de filtros creciente (64 → 128 → 256 → 512 → 512)
   - Cada bloque usa convoluciones 3×3 con padding `same` y activaciones ReLU, seguidas de max pooling 2×2
   - La salida aplanada alimenta dos capas densas de 4.096 unidades, seguidas de una capa de salida softmax de 2 unidades
3. **Entrenamiento:** Compilado con `Adam (lr=0.001)` y pérdida `categorical_crossentropy`. Uso de `EarlyStopping` (paciencia=3) sobre la exactitud de validación y `ModelCheckpoint` para guardar los mejores pesos.
4. **Inferencia:** Carga del modelo guardado para ejecutar predicciones en imágenes individuales, devolviendo la clase con mayor probabilidad softmax.

## Resultados

El modelo fue entrenado durante 3 épocas sobre un dataset local pequeño de **18 imágenes** (16 entrenamiento, 2 validación). La exactitud de validación se estabilizó en **50%** en todas las épocas — esencialmente azar — lo cual es esperable dado el tamaño del dataset. La parada anticipada se activó al terminar la época 3 y restauró los mejores pesos de la época 1. La pérdida de entrenamiento sí disminuyó de forma constante (0,76 → 0,69), confirmando que el modelo estaba aprendiendo, pero el conjunto de validación era demasiado pequeño para mostrar una mejora significativa.

El valor principal de este proyecto radica en la implementación arquitectónica: construir y entrenar una CNN completa estilo VGG16 de extremo a extremo. Volver a ejecutar con el dataset completo de Perros vs. Gatos de Kaggle (25.000 imágenes) sería el paso natural siguiente.

## Tecnologías utilizadas

`Python` · `TensorFlow` · `Keras` · `NumPy` · `Matplotlib`

## Ejecución local

```bash
git clone https://github.com/matthewkane-ml/ML_DeepLearning_MTK.git
cd ML_DeepLearning_MTK
pip install -r requirements.txt

# Organiza tus imágenes en:
# data/raw/dog/   (imágenes de perros)
# data/raw/cat/   (imágenes de gatos)

# Luego abre y ejecuta el notebook
jupyter notebook src/DeepLearning-VGG16.ipynb
```

## Próximos pasos

- Usar **transfer learning** con los pesos preentrenados de VGG16 en ImageNet en lugar de entrenar desde cero — esto mejoraría drásticamente la exactitud con un dataset pequeño
- Añadir **aumento de datos** (volteos horizontales, rotaciones, zoom) para reducir el sobreajuste
- Entrenar más épocas con un programa de tasa de aprendizaje decreciente para permitir una convergencia más completa del modelo
- Extender a un clasificador multiclase más allá del binario perro/gato

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
