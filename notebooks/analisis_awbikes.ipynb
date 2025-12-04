# ============================================
# CÓDIGO COMPLETO - ANÁLISIS Y GRÁFICAS AW BIKES
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Configuración de estilo profesional
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

# =============================================================================
# CARGA DEL ARCHIVO DEPURADO
# =============================================================================
print("=" * 80)
print("CARGANDO DATOS...")
print("=" * 80)

# Si estás en Google Colab, carga así:
# from google.colab import files
# files.upload()  # Selecciona tu archivo

# Si estás en local, asegúrate que el archivo esté en la carpeta
file_path = "datos-actividad-depurada.xlsx"

try:
    df = pd.read_excel(file_path)
    print(f"\n✓ Archivo cargado exitosamente")
    print(f"Dimensiones: {df.shape}")
    print(f"\nColumnas disponibles:\n{df.columns.tolist()}")
except Exception as e:
    print(f"❌ Error cargando archivo: {e}")
    print("Asegúrate de que 'datos-actividad-depurada.xlsx' esté en la carpeta actual o en Google Drive")


# =============================================================================
# EXPLORACIÓN INICIAL
# =============================================================================
print("\n" + "=" * 80)
print("EXPLORACIÓN INICIAL DE DATOS")
print("=" * 80)
print(f"\nPrimeras 5 filas:\n{df.head()}")
print(f"\nInfo del dataset:\n{df.info()}")
print(f"\nValores nulos:\n{df.isnull().sum()}")

# Identificar columnas numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nColumnas numéricas: {numeric_cols}")


# =============================================================================
# GRÁFICA 1: MATRIZ DE CORRELACIÓN (HEATMAP)
# =============================================================================
print("\n" + "=" * 80)
print("GENERANDO GRÁFICA 1: MATRIZ DE CORRELACIÓN")
print("=" * 80)

# Identificar variable objetivo (BikeBuyer o similar)
target_col = None
for col in df.columns:
    if 'bike' in col.lower() or 'buyer' in col.lower():
        target_col = col
        break

if target_col is None and len(numeric_cols) > 0:
    # Si no hay columna explícita, asumir la última como target
    target_col = numeric_cols[-1]

print(f"\nVariable objetivo detectada: {target_col}")

# Crear matriz de correlación solo con variables numéricas
if len(numeric_cols) > 1:
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación - Variables Numéricas (AW-Bikes)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('matriz_correlacion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Gráfica de correlación generada y guardada como 'matriz_correlacion.png'")
    
    # Análisis de correlaciones con variable objetivo
    if target_col in corr_matrix.columns:
        print(f"\nCorrelaciones con '{target_col}':")
        correlaciones = corr_matrix[target_col].sort_values(ascending=False)
        print(correlaciones)


# =============================================================================
# GRÁFICA 2: MATRIZ DE CONFUSIÓN (DATOS REALES DE TU ACTIVIDAD)
# =============================================================================
print("\n" + "=" * 80)
print("GENERANDO GRÁFICA 2: MATRIZ DE CONFUSIÓN - ÁRBOL DE DECISIÓN")
print("=" * 80)

# Usar los datos reales de tu archivo MATRICES_CONFUSION.xlsx
# Estructura: VN=1581, FP=888, FN=514, VP=2524
matriz_datos = np.array([[1581, 888], 
                         [514, 2524]])

etiquetas = ['No Compra\n(Real)', 'Compra\n(Real)']
categorias = ['No Compra\n(Predicho)', 'Compra\n(Predicho)']

plt.figure(figsize=(9, 7))
sns.heatmap(matriz_datos, annot=True, fmt='d', cmap='Blues', 
            xticklabels=categorias, yticklabels=etiquetas, 
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"label": "Cantidad de casos"},
            linewidths=2, linecolor='black')

plt.title('Matriz de Confusión - Árbol de Decisión\n(Conjunto de Prueba: 5,507 clientes)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicción del Modelo', fontsize=12, fontweight='bold')
plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=300, bbox_inches='tight')
plt.show()

# Cálculo de métricas
VN, FP = matriz_datos[0]
FN, VP = matriz_datos[1]
total = VN + FP + FN + VP

accuracy = (VP + VN) / total
precision = VP / (VP + FP) if (VP + FP) > 0 else 0
recall = VP / (VP + FN) if (VP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = VN / (VN + FP) if (VN + FP) > 0 else 0

print("\n✓ Matriz de Confusión generada")
print(f"\nMÉTRICAS DE DESEMPEÑO:")
print(f"  • Accuracy (Exactitud):     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  • Precision (Precisión):    {precision:.4f} ({precision*100:.2f}%)")
print(f"  • Recall (Sensibilidad):    {recall:.4f} ({recall*100:.2f}%)")
print(f"  • F1-Score:                 {f1:.4f}")
print(f"  • Specificity (Especificidad): {specificity:.4f} ({specificity*100:.2f}%)")

print(f"\nINTERPRETACIÓN:")
print(f"  • Verdaderos Positivos (VP):   {VP} - Compradores correctamente identificados")
print(f"  • Verdaderos Negativos (VN):   {VN} - No compradores correctamente identificados")
print(f"  • Falsos Positivos (FP):       {FP} - No compradores erróneamente clasificados como compradores")
print(f"  • Falsos Negativos (FN):       {FN} - Compradores erróneamente clasificados como no compradores")


# =============================================================================
# GRÁFICA 3: DISTRIBUCIÓN DE PROBABILIDADES DE COMPRA
# =============================================================================
print("\n" + "=" * 80)
print("GENERANDO GRÁFICA 3: DISTRIBUCIÓN DE PROBABILIDADES")
print("=" * 80)

try:
    # Intentar cargar las predicciones
    df_pred = pd.read_excel("Predicciones_arbol_decision_FINAL.xlsx")
    
    print(f"\n✓ Archivo de predicciones cargado")
    print(f"Columnas: {df_pred.columns.tolist()}")
    
    # Buscar columna de probabilidad
    prob_col = None
    for col in df_pred.columns:
        if 'prob' in col.lower() or 'probabilidad' in col.lower():
            prob_col = col
            break
    
    if prob_col is None:
        print("⚠ No se encontró columna de probabilidad. Usando la última columna numérica.")
        prob_col = df_pred.select_dtypes(include=[np.number]).columns[-1]
    
    print(f"Columna de probabilidad: {prob_col}")
    
    # Crear histograma
    plt.figure(figsize=(12, 7))
    
    sns.histplot(data=df_pred, x=prob_col, bins=30, kde=True, 
                 color='green', alpha=0.6, edgecolor='black', linewidth=0.5)
    
    # Líneas de umbral
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2.5, 
                label='Umbral Estándar (0.5)', alpha=0.8)
    plt.axvline(0.7, color='orange', linestyle='--', linewidth=2.5, 
                label='Umbral Premium (0.7)', alpha=0.8)
    
    # Estadísticas en la gráfica
    media = df_pred[prob_col].mean()
    mediana = df_pred[prob_col].median()
    plt.axvline(media, color='blue', linestyle=':', linewidth=2, 
                label=f'Media: {media:.3f}', alpha=0.8)
    
    plt.title('Distribución de Probabilidades de Compra\n(Árbol de Decisión)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Probabilidad de Compra (0 a 1)', fontsize=12, fontweight='bold')
    plt.ylabel('Cantidad de Clientes', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distribucion_probabilidades.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Distribución de probabilidades generada")
    print(f"\nESTADÍSTICAS DE PROBABILIDAD:")
    print(f"  • Media:        {media:.4f}")
    print(f"  • Mediana:      {mediana:.4f}")
    print(f"  • Mín:          {df_pred[prob_col].min():.4f}")
    print(f"  • Máx:          {df_pred[prob_col].max():.4f}")
    print(f"  • Desv. Est.:   {df_pred[prob_col].std():.4f}")
    
    # Contar clientes por segmento
    seg_alto = (df_pred[prob_col] > 0.7).sum()
    seg_medio = ((df_pred[prob_col] >= 0.5) & (df_pred[prob_col] <= 0.7)).sum()
    seg_bajo = (df_pred[prob_col] < 0.5).sum()
    
    print(f"\nSEGMENTACIÓN POR PROBABILIDAD:")
    print(f"  • Segmento Alto (P > 0.7):        {seg_alto} clientes ({seg_alto/len(df_pred)*100:.2f}%)")
    print(f"  • Segmento Medio (0.5 ≤ P ≤ 0.7): {seg_medio} clientes ({seg_medio/len(df_pred)*100:.2f}%)")
    print(f"  • Segmento Bajo (P < 0.5):       {seg_bajo} clientes ({seg_bajo/len(df_pred)*100:.2f}%)")

except FileNotFoundError:
    print("⚠ No se encontró 'Predicciones_arbol_decision_FINAL.xlsx'")
    print("  Generando distribución simulada...")
    
    # Crear datos simulados realistas
    np.random.seed(42)
    prob_simulada = np.concatenate([
        np.random.beta(2, 5, 1500),  # Clientes con baja probabilidad
        np.random.beta(5, 2, 1000)   # Clientes con alta probabilidad
    ])
    
    plt.figure(figsize=(12, 7))
    plt.hist(prob_simulada, bins=30, color='green', alpha=0.6, edgecolor='black', linewidth=0.5)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2.5, label='Umbral (0.5)', alpha=0.8)
    plt.axvline(0.7, color='orange', linestyle='--', linewidth=2.5, label='Umbral Premium (0.7)', alpha=0.8)
    plt.title('Distribución de Probabilidades de Compra (Simulada)', fontsize=16, fontweight='bold')
    plt.xlabel('Probabilidad de Compra (0 a 1)', fontsize=12, fontweight='bold')
    plt.ylabel('Cantidad de Clientes', fontsize=12, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribucion_probabilidades_simulada.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Distribución simulada generada y guardada")


# =============================================================================
# GRÁFICA 4: BOXPLOT DE VARIABLES NUMÉRICAS POR VARIABLE OBJETIVO
# =============================================================================
print("\n" + "=" * 80)
print("GENERANDO GRÁFICA 4: BOXPLOTS POR VARIABLE OBJETIVO")
print("=" * 80)

if target_col and target_col in df.columns:
    # Seleccionar top 4 variables numéricas (excluyendo target)
    top_vars = [col for col in numeric_cols if col != target_col][:4]
    
    if len(top_vars) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, var in enumerate(top_vars):
            sns.boxplot(data=df, x=target_col, y=var, ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'{var} por {target_col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(target_col, fontsize=10)
            axes[idx].set_ylabel(var, fontsize=10)
        
        plt.suptitle('Distribución de Variables Numéricas por Variable Objetivo', 
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('boxplots_variables.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Boxplots generados y guardados")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN DE GRÁFICAS GENERADAS")
print("=" * 80)
print("""
✓ 1. matriz_correlacion.png
   - Heatmap de correlaciones entre variables numéricas
   - Ayuda a identificar predictores relevantes

✓ 2. matriz_confusion.png
   - Visualización de VP, VN, FP, FN
   - Métricas de desempeño del Árbol de Decisión

✓ 3. distribucion_probabilidades.png
   - Histograma de probabilidades de compra
   - Segmentación por umbral de probabilidad

✓ 4. boxplots_variables.png (opcional)
   - Distribución de variables numéricas por grupo de compra

PRÓXIMOS PASOS:
1. Inserta estas imágenes en tu informe ejecutivo de Word
2. Referencia cada gráfica en el análisis de criterios
3. Asegúrate de que el tamaño sea legible en el documento final
""")
