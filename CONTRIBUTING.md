# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto **Modelo Predictivo AW-Bikes (II)**!

## 📋 Cómo Contribuir

### 1. Fork el Repositorio

```bash
# En GitHub, click en "Fork" en la esquina superior derecha
```

### 2. Clona tu Fork Local

```bash
git clone https://github.com/TU_USUARIO/Mini-Case-AW-Bikes-II.git
cd Mini-Case-AW-Bikes-II
```

### 3. Crea una Rama para tu Contribución

```bash
git checkout -b feature/mi-contribucion
# O para bugfixes:
git checkout -b bugfix/correccion-importante
```

### 4. Realiza tus Cambios

```bash
# Edita los archivos necesarios
# Asegúrate de:
# - Mantener el estilo de código consistente
# - Usar nombres descriptivos
# - Comentar el código complejo
# - Actualizar la documentación si es necesario
```

### 5. Commit tus Cambios

```bash
git add .
git commit -m "Descripción clara y concisa de los cambios

- Primer cambio importante
- Segundo cambio importante
- Etc."
```

### 6. Push a tu Fork

```bash
git push origin feature/mi-contribucion
```

### 7. Crea un Pull Request (PR)

- Ve a GitHub y verás un botón "Compare & pull request"
- Describe claramente qué cambios hiciste y por qué
- Incluye referencias a issues si es aplicable
- Espera a que se revise

---

## 🎯 Tipos de Contribuciones Bienvenidas

### 📝 Documentación
- Mejorar README
- Clarificar ejemplos
- Traducir documentación
- Añadir guías de uso

### 🐛 Bug Fixes
- Reportar bugs con claridad
- Proponer soluciones
- Incluir pasos para reproducir

### ✨ Nuevas Características
- Mejoras al modelo (nuevos algoritmos, ajustes)
- Nuevas visualizaciones
- Optimizaciones de rendimiento
- Extensiones funcionales

### 🧪 Mejoras de Testing
- Escribir tests unitarios
- Aumentar cobertura de testing
- Validación de resultados

### 📊 Análisis Adicionales
- Análisis de nuevas variables
- Comparación con otros datasets
- Estudios de casos complementarios

---

## 📋 Estándares de Código

### Python PEP 8
```python
# ✅ BUENO
def calcular_importancia_variables(modelo, X_test):
    """Calcula la importancia de cada variable en el modelo."""
    importancias = modelo.feature_importances_
    return importancias

# ❌ MALO
def calc_imp(m, x):
    i = m.feature_importances_
    return i
```

### Docstrings
```python
def entrenar_modelo(X_train, y_train, profundidad=5):
    """
    Entrena un Árbol de Decisión con los parámetros especificados.
    
    Args:
        X_train (DataFrame): Features de entrenamiento
        y_train (Series): Target de entrenamiento
        profundidad (int): Profundidad máxima del árbol
    
    Returns:
        DecisionTreeClassifier: Modelo entrenado
    
    Example:
        >>> modelo = entrenar_modelo(X_train, y_train, profundidad=5)
        >>> print(modelo.score(X_test, y_test))
    """
    from sklearn.tree import DecisionTreeClassifier
    
    modelo = DecisionTreeClassifier(max_depth=profundidad, random_state=123)
    modelo.fit(X_train, y_train)
    return modelo
```

### Comentarios
```python
# Calcular correlaciones entre predictores
correlation_matrix = X.corr()

# Identificar correlaciones fuertes (r > 0.7)
strong_corr = correlation_matrix[correlation_matrix > 0.7].dropna(how='all')
```

---

## 🧪 Testing

### Ejecutar Tests
```bash
pytest tests/
pytest --cov=codigo tests/  # Con cobertura
```

### Escribir Tests
```python
# tests/test_modelo.py
import pytest
from codigo.modelo_arbol_decision import entrenar_modelo

def test_entrenar_modelo():
    """Verifica que el modelo se entrena correctamente."""
    X_train, y_train = [[0, 1], [1, 0]], [0, 1]
    modelo = entrenar_modelo(X_train, y_train)
    
    assert modelo is not None
    assert hasattr(modelo, 'predict')

def test_predicciones_validas():
    """Verifica que las predicciones son válidas."""
    X_test = [[0, 1]]
    modelo = entrenar_modelo([[0, 1], [1, 0]], [0, 1])
    predicciones = modelo.predict(X_test)
    
    assert predicciones[0] in [0, 1]
```

---

## 📚 Citación de Referencias

Si añades nuevos métodos o teoría, incluye referencias académicas:

```python
"""
Implementación de validación cruzada estratificada.

Referencias:
    Hastie, T., Tibshirani, R., & Friedman, J. (2009).
    The elements of statistical learning (2nd ed.). Springer.
"""
```

---

## 🔄 Proceso de Revisión

1. **Automático:** Se ejecutan tests automáticos en tu PR
2. **Manual:** Un revisor verifica tu código
3. **Feedback:** Se pueden solicitar cambios
4. **Aprobación:** Una vez aprobado, se fusiona (merge)

---

## 📝 Commit Messages

### Formato
```
[TYPE] Descripción breve (máx 50 caracteres)

Descripción detallada si es necesaria.
Puede incluir múltiples párrafos.

Fixes #123
Related to #456
```

### Tipos
- `feat`: Nueva característica
- `fix`: Corrección de bug
- `docs`: Cambios de documentación
- `style`: Formato, indentación (sin cambios funcionales)
- `refactor`: Refactorización de código
- `test`: Añadir o mejorar tests
- `perf`: Mejora de rendimiento

### Ejemplos
```
feat: Añadir análisis de importancia de variables

Implementa método feature_importance en DecisionTree.
Calcula impacto relativo de cada predictor.

Fixes #45

---

fix: Corregir error en lectura de Excel

El error ocurría cuando había valores nulos.
Ahora se manejan correctamente.

Related to #32

---

docs: Mejorar README con ejemplos de uso

Añade secciones de: instalación, uso rápido, troubleshooting.
```

---

## 🚀 Workflow Completo (Ejemplo)

```bash
# 1. Fork en GitHub
# 2. Clonar
git clone https://github.com/TU_USUARIO/Mini-Case-AW-Bikes-II.git
cd Mini-Case-AW-Bikes-II

# 3. Crear rama
git checkout -b feat/nueva-visualizacion

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Hacer cambios
# Editar archivos...

# 6. Probar localmente
python -m pytest tests/

# 7. Commit
git add .
git commit -m "feat: Añadir gráfica de importancia de variables

Implementa visualización mejorada del feature importance.
Usa colores diferenciados por rango de importancia."

# 8. Push
git push origin feat/nueva-visualizacion

# 9. Crear PR en GitHub
# - Ir a: https://github.com/TU_USUARIO/Mini-Case-AW-Bikes-II
# - Click en "Compare & pull request"
# - Completar descripción
# - Submit
```

---

## ❓ Preguntas Frecuentes

**P: ¿Puedo cambiar directamente en main?**
R: No. Siempre crea una rama nueva para tus cambios.

**P: ¿Cuánto tiempo tarda la revisión?**
R: Típicamente 2-5 días. Depende de la complejidad.

**P: ¿Qué si mi PR no es aceptado?**
R: Se proporcionará feedback constructivo. Puedes ajustar y reenviar.

**P: ¿Necesito tests para todo?**
R: Idealmente sí, pero si no, los revisores pueden ayudar.

**P: ¿Puedo contribuir desde Google Colab?**
R: Sí, pero es más fácil desde una máquina local con Git instalado.

---

## 📞 Contacto

- **Issues:** Usa GitHub Issues para bugs/features
- **Discusiones:** Para preguntas generales
- **Email:** [Tu email]

---

## ⚖️ Licencia

Al contribuir, aceptas que tu código se distribuya bajo la licencia MIT del proyecto.

---

**¡Gracias por contribuir! 🎉**
