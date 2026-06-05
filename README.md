# 🚀 Custom CNN Image Classifier Pipeline with PyTorch

Clasificador de imágenes modular y escalable construido sobre **PyTorch** y **Torchvision**. Este proyecto implementa una arquitectura limpia basada en el principio de responsabilidad única, separando el modelo, el motor de entrenamiento (`Trainer`) y el pipeline de datos para facilitar la experimentación con diferentes arquitecturas (ej. ResNet18) y optimizadores.

---

## 📌 Características Clave

* **Diseño 100% Modular:** Componentes totalmente desacoplados (Modelo, Trainer, Data Pipeline).
* **Inyección de Dependencias:** El `Trainer` es agnóstico al modelo, optimizador o scheduler que utilices.
* **Transfer Learning Eficiente:** Configurado como extractor de características utilizando pesos congelados de ResNet18 de forma predeterminada.
* **Ajuste Dinámico de LR:** Integración nativa con schedulers cíclicos (`CyclicLR`) para evitar estancamientos en mínimos locales.
* **Evaluación Automatizada:** Monitoreo de pérdidas y precisión por época, restaurando automáticamente los mejores pesos al finalizar el entrenamiento.

---

## 📁 Estructura del Proyecto

```text
├── data/                  # Datasets de entrenamiento y validación
│   ├── train/
│   └── val/
├── src/                   # Código fuente modular
│   ├── __init__.py
│   ├── models.py          # Definición de arquitecturas de Redes Neuronales
│   ├── trainer.py         # Orquestador del loop de entrenamiento y validación
│   └── utils.py           # Funciones de ayuda (guardado, transformaciones, etc.)
├── outputs/               # Modelos exportados (.pt) y gráficos de métricas
├── main.py                # Script principal de ejecución (Orquestador)
└── README.md
