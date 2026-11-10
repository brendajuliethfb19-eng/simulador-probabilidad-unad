"""Simulador Interactivo de Variables Aleatorias - UNAD"""
import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import List, Tuple
 
 
# --- ESTILOS CSS PROFESIONALES ---
st.set_page_config(
    page_title="Simulador de Variables Aleatorias",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)
 
st.markdown(
    """
<style>
    :root {
        --primary: #2c3e50;
        --secondary: #3498db;
        --accent: #e74c3c;
        --success: #27ae60;
        --gray: #f8f9fa;
        --dark-gray: #6c757d;
    }
    .main {
        padding: 1rem 2rem;
        max-width: 1200px;
    }
    .stTitle {
        color: var(--primary) !important;
    }
    .stSidebar {
        background: var(--gray);
        border-right: 1px solid #dee2e6;
    }
    .stSelectbox > div > div > div,
    .stSlider > div > div > div > div {
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }
    .stSlider > div > div > div > div {
        background: var(--secondary);
    }
    h1, h2, h3 {
        color: var(--primary) !important;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stNumberInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #ced4da;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .stExpander > summary {
        font-weight: 500;
        color: var(--primary);
        padding: 0.5rem 1rem;
    }
    .stButton > button {
        background: var(--secondary);
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #2980b9;
    }
    .formula {
        background: #f1f3f5;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        font-family: 'Monospace', monospace;
        color: var(--primary);
    }
</style>
    """,
    unsafe_allow_html=True,
)
 
 
# --- FUNCIONES AUXILIARES ---
def _combinacion(n: int, k: int) -> int:
    """Calcula el binomio C(n, k)."""
    return math.comb(n, k)
 
 
def _pmf_binomial(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """PMF de distribución binomial."""
    return np.array([
        _combinacion(n, int(i)) * (p ** i) * ((1 - p) ** (n - i)) 
        for i in k
    ])
 
 
def _pmf_poisson(k: np.ndarray, mu: float) -> np.ndarray:
    """PMF de distribución de Poisson."""
    return np.array([
        (np.exp(-mu) * (mu ** i)) / math.factorial(int(i)) 
        for i in k
    ])
 
 
def _pdf_normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """PDF de distribución normal."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
 
 
def _pdf_exponencial(x: np.ndarray, lam: float) -> np.ndarray:
    """PDF de distribución exponencial."""
    return lam * np.exp(-lam * x)
 
 
# --- FUNCIONES DE GRAFICO ---
def graficar_distribucion_discreta(x: np.ndarray, y: np.ndarray, titulo: str):
    """Grafica una distribución discreta."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        x, y,
        color="#3498db",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xlabel("Valores de la Variable (x)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Probabilidad P(X = x)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(titulo, fontsize=12, color="#2c3e50", fontweight="bold")
    st.pyplot(fig)
 
 
def graficar_distribucion_continua(x: np.ndarray, y: np.ndarray, titulo: str):
    """Grafica una distribución continua."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="#e74c3c", linewidth=2.5, label="f(x)")
    ax.fill_between(x, y, color="#e74c3c", alpha=0.15)
    ax.set_xlabel("Valores (x)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Densidad f(x)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    ax.set_title(titulo, fontsize=12, color="#2c3e50", fontweight="bold")
    st.pyplot(fig)
 
 
# --- FORMULARIO DE ENTRADA DE PARÁMETROS ---
def formulario_parametros_discretos():
    """Formulario para parámetros de distribuciones discretas."""
    st.subheader("🔢 Parámetros de la Distribución")
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input(
            "Número de ensayos (n)",
            min_value=1,
            value=10,
            step=1,
            help="Total de ensayos independientes (solo distribuciones discretas).",
        )
    with col2:
        p = st.number_input(
            "Probabilidad de éxito (p)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Probabilidad de éxito en cada ensayo (0 a 1).",
        )
    return n, p
 
 
def formulario_parametros_poisson():
    """Formulario para parámetros de Poisson."""
    st.subheader("📊 Parámetros de Poisson")
    mu = st.number_input(
        "Tasa media (μ)",
        min_value=0.01,
        value=5.0,
        step=0.1,
        help="Número promedio de eventos en un intervalo.",
    )
    return mu
 
 
def formulario_parametros_normales():
    """Formulario para parámetros normales."""
    st.subheader("📈 Parámetros de la Distribución Normal")
    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input(
            "Media (μ)",
            value=0.0,
            step=0.1,
        )
    with col2:
        sigma = st.number_input(
            "Desviación estándar (σ)",
            min_value=0.1,
            value=1.5,
            step=0.1,
        )
    return mu, sigma
 
 
def formulario_parametros_exponencial():
    """Formulario para parámetros exponenciales."""
    st.subheader("⏱️ Parámetros de la Distribución Exponencial")
    lam = st.number_input(
        "Parámetro de tasa (λ)",
        min_value=0.1,
        value=1.0,
        step=0.1,
        help="Lambda > 0 es la tasa de ocurrencia.",
    )
    return lam
 
 
# --- CÁLCULOS MATEMÁTICOS ---
def calcular_estadisticas_discretas(tipo: str, n: int = None, p: float = None, mu: float = None):
    """Calcula estadísticas para distribuciones discretas."""
    if tipo == "binomial":
        media = n * p
        varianza = n * p * (1 - p)
        desv = np.sqrt(varianza)
        return {"media": media, "varianza": varianza, "desv": desv}
    elif tipo == "poisson":
        media = mu
        varianza = mu
        desv = np.sqrt(mu)
        return {"media": media, "varianza": varianza, "desv": desv}
    return {}
 
 
def calcular_estadisticas_continuas(tipo: str, sigma: float = None, lam: float = None):
    """Calcula estadísticas para distribuciones continuas."""
    if tipo == "normal":
        media = 0  # Se calculará del mu
        varianza = sigma ** 2
        desv = sigma
        return {"media": "μ", "varianza": varianza, "desv": desv}
    elif tipo == "exponencial":
        media = 1 / lam
        varianza = 1 / (lam ** 2)
        desv = 1 / lam
        return {"media": media, "varianza": varianza, "desv": desv}
    return {}
 
 
# --- MÓDULO DE VALIDACIÓN ---
def modulo_validacion(tipo: str, distribucion: str, n: int = None, p: float = None, 
                      mu: float = None, sigma: float = None, lam: float = None,
                      val_x: float = None):
    """Muestra el módulo de validación manual."""
    st.markdown("---")
    st.markdown("### 🔍 Validación Manual de Ejercicio")
    if distribucion == "Binomial" and val_x is not None:
        comb_val = _combinacion(n, int(val_x))
        prob = comb_val * (p ** val_x) * ((1 - p) ** (n - val_x))
        with st.expander("Ver cálculo detallado", expanded=True):
            st.info(
                f"**Binomial (n={n}, p={p}, x={int(val_x)}):**\n\n"
                f"1. Combinatoria $\\binom{{{n}}}{{{int(val_x)}}}$: {comb_val}\n"
                f"2. Término éxito ($p^{val_x}$): {p**val_x:.5f}\n"
                f"3. Término fracaso ($(1-p)^{n-val_x}$): {((1-p)**(n-val_x)):.5f}\n\n"
                f"👉 **P(X = {int(val_x)}) = {prob:.6f}** "
                f"({prob*100:.2f}%)"
            )
    elif distribucion == "Poisson" and val_x is not None:
        prob = (np.exp(-mu) * (mu ** val_x)) / math.factorial(int(val_x))
        with st.expander("Ver cálculo detallado", expanded=True):
            st.info(
                f"**Poisson (μ={mu}, x={int(val_x)}):**\n\n"
                f"1. Factor $e^{{-{mu}}}$: {np.exp(-mu):.5f}\n"
                f"2. Término tasa ($\\mu^{val_x}$): {mu**val_x:.5f}\n"
                f"3. Factorial ($x!$): {math.factorial(int(val_x))}\n\n"
                f"👉 **P(X = {int(val_x)}) = {prob:.6f}** "
                f"({prob*100:.2f}%)"
            )
    elif distribucion == "Normal" and val_x is not None:
        from scipy.stats import norm
        dens = norm.pdf(val_x, loc=0, scale=sigma)  # Placeholder
        st.info(
            f"**Normal (μ=0, σ={sigma}, x={val_x:.2f}):**\n\n"
            f"👉 **f(x) = {dens:.6f}**"
        )
    elif distribucion == "Exponencial" and val_x is not None:
        prob = lam * np.exp(-lam * val_x)
        st.info(
            f"**Exponencial (λ={lam}, x={val_x:.2f}):**\n\n"
            f"👉 **f(x) = {prob:.6f}**"
        )
 
 
# --- INTERFAZ PRINCIPAL ---
st.title("📊 Simulador de Variables Aleatorias")
st.markdown(
    "Universidad Nacional Abierta y a Distancia (UNAD) — Herramienta interactiva"
    " para la exploración y validación de variables aleatorias en ejercicios"
    " de probabilidad y estadística."
)
st.markdown("---")
 
# Selector de tipo de variable
tipo_variable = st.sidebar.selectbox(
    "Tipo de Variable Aleatoria",
    ["Discreta", "Continua"],
    help="Seleccione el tipo de variable para configurar el simulador.",
)
 
# Área principal basada en el tipo
if tipo_variable == "Discreta":
    st.sidebar.header("⚙️ Configuración")
    # Detectar distribución
    distribucion = st.sidebar.radio(
        "Modelo de Distribución",
        ["Binomial", "Poisson"],
        help="Seleccione el modelo de distribución discreta.",
    )
    # Formulario de parámetros
    if distribucion == "Binomial":
        n, p = formulario_parametros_discretos()
        stats = calcular_estadisticas_discretas("binomial", n=n, p=p)
        x = np.arange(0, n + 1)
        y = _pmf_binomial(x, n, p)
        st.subheader(f"📌 Distribución Binomial: X ~ Bin(n={n}, p={p})")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{stats['media']:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")
        graficar_distribucion_discreta(x, y, f"Distribución Binomial n={n}, p={p}")
        # Validación
        val_x = st.number_input(
            "Ingrese valor de x para validar:",
            min_value=0,
            max_value=int(n),
            value=0,
            step=1,
            help="Digite un valor de x dentro del rango [0, n] para validar.",
        )
        modulo_validacion("discreta", "Binomial", n=n, p=p, val_x=val_x)
    else:  # Poisson
        mu = formulario_parametros_poisson()
        stats = calcular_estadisticas_discretas("poisson", mu=mu)
        x_max = int(mu * 3) + 15
        x = np.arange(0, x_max)
        y = _pmf_poisson(x, mu)
        st.subheader(f"📌 Distribución de Poisson: X ~ Pois(μ={mu})")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{stats['media']:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")
        graficar_distribucion_discreta(x, y, f"Distribución de Poisson μ={mu}")
        # Validación
        val_x = st.number_input(
            "Ingrese valor de x para validar:",
            min_value=0,
            max_value=int(mu * 3),
            value=int(mu),
            step=1,
            help="Digite un valor de x para validar el cálculo.",
        )
        modulo_validacion("discreta", "Poisson", mu=mu, val_x=val_x)
 
else:  # Continua
    st.sidebar.header("⚙️ Configuración")
    distribucion = st.sidebar.radio(
        "Modelo de Distribución",
        ["Normal (Gaussiana)", "Exponencial"],
        help="Seleccione el modelo de distribución continua.",
    )
    # Formulario de parámetros
    if distribucion == "Normal (Gaussiana)":
        mu, sigma = formulario_parametros_normales()
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        y = _pdf_normal(x, mu, sigma)
        st.subheader(f"📌 Distribución Normal: X ~ N(μ={mu}, σ²={sigma**2})")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{mu:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{sigma**2:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{sigma:.4f}")
        graficar_distribucion_continua(x, y, f"Distribución Normal μ={mu}, σ={sigma}")
        # Validación
        val_x = st.number_input(
            "Ingrese valor de x para evaluar densidad:",
            value=mu,
            step=0.1,
            help="Digite un valor de x para evaluar la función de densidad.",
        )
        modulo_validacion("continua", "Normal", sigma=sigma, val_x=val_x)
    else:  # Exponencial
        lam = formulario_parametros_exponencial()
        x = np.linspace(0, 5 / lam, 300)
        y = _pdf_exponencial(x, lam)
        stats = calcular_estadisticas_continuas("exponencial", lam=lam)
        st.subheader(f"📌 Distribución Exponencial: X ~ Exp(λ={lam})")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{stats['media']:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")
        graficar_distribucion_continua(x, y, f"Distribución Exponencial λ={lam}")
        # Validación
        val_x = st.number_input(
            "Ingrese valor de x para evaluar densidad:",
            value=1/lam,
            step=0.1,
            help="Digite un valor de x ≥ 0 para evaluar la función de densidad.",
        )
        modulo_validacion("continua", "Exponencial", lam=lam, val_x=val_x)