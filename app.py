"""Simulador Interactivo de Variables Aleatorias - UNAD"""
import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import List, Tuple
 
 
# ==========================================
# COLORES DEFINIDOS COMO VARIABLES (SIN F-STRINGS EN CSS)
# ==========================================
COLOR_PRIMARY_DARK = "#1B365D"
COLOR_PRIMARY_MEDIUM = "#005A9C"
COLOR_PRIMARY_LIGHT = "#60C1E5"
COLOR_PRIMARY_VERY_LIGHT = "#A0E0EF"
COLOR_BACKGROUND_ALT = "#F0F4F8"
 
 
# ==========================================
# CONFIGURACIÓN DE PÁGINA - CSS SIN F-STRINGS
# ==========================================
st.set_page_config(
    page_title="Simulador de Variables Aleatorias",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)
 
# CSS CON COLORES HARDCODEADOS (SIN F-STRINGS)
st.markdown(
    """
<style>
    :root {
        --primary-dark: #1B365D;
        --primary-medium: #005A9C;
        --primary-light: #60C1E5;
        --primary-very-light: #A0E0EF;
        --background-alt: #F0F4F8;
    }
    .main {
        padding: 1rem 2rem;
        max-width: 1200px;
        background: #A0E0EF;
    }
    .stTitle {
        color: #1B365D !important;
    }
    .stSidebar {
        background: #A0E0EF;
        border-right: 2px solid #005A9C;
    }
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div > div {
        border-radius: 6px;
        border: 1px solid #ced4da;
        background: white;
    }
    .stSlider > div > div > div > div {
        background: #005A9C;
    }
    h1, h2, h3 {
        color: #1B365D !important;
        font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        margin: 0.5rem 0;
        background: white;
    }
    .stExpander > summary {
        font-weight: 500;
        color: #1B365D;
        padding: 0.5rem 1rem;
    }
    .stButton > button {
        background: #005A9C;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
        width: 100%;
    }
    .stButton > button:hover {
        background: #1B365D;
    }
    .formula {
        background: #e8f0f7;
        border-left: 4px solid #005A9C;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-family: 'Monospace', monospace;
        color: #1B365D;
    }
</style>
    """,
    unsafe_allow_html=True,
)
 
 
# ==========================================
# Resto del código igual que antes (funciones matemáticas, gráficos, etc.)
# ==========================================
 
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
    """PDF de distribución normal usando numpy."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
 
 
def _pdf_exponencial(x: np.ndarray, lam: float) -> np.ndarray:
    """PDF de distribución exponencial."""
    return lam * np.exp(-lam * x)
 
 
def graficar_distribucion_discreta(x: np.ndarray, y: np.ndarray, titulo: str):
    """Grafica una distribución discreta."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        x, y,
        color="#005A9C",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        width=0.6,
    )
    ax.set_xlabel("Valores de la Variable (x)", fontsize=10, fontweight="bold", color="#1B365D")
    ax.set_ylabel("Probabilidad P(X = x)", fontsize=10, fontweight="bold", color="#1B365D")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(titulo, fontsize=12, color="#1B365D", fontweight="bold")
    ax.tick_params(axis='x', colors="#1B365D")
    ax.tick_params(axis='y', colors="#1B365D")
    st.pyplot(fig)
 
 
def graficar_distribucion_continua(x: np.ndarray, y: np.ndarray, titulo: str):
    """Grafica una distribución continua."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="#1B365D", linewidth=2.5, label="f(x)")
    ax.fill_between(x, y, color="#60C1E5", alpha=0.3)
    ax.set_xlabel("Valores (x)", fontsize=10, fontweight="bold", color="#1B365D")
    ax.set_ylabel("Densidad f(x)", fontsize=10, fontweight="bold", color="#1B365D")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend()
    ax.set_title(titulo, fontsize=12, color="#1B365D", fontweight="bold")
    ax.tick_params(axis='x', colors="#1B365D")
    ax.tick_params(axis='y', colors="#1B365D")
    st.pyplot(fig)
 
 
def formulario_parametros_discretos():
    st.subheader("🔢 Parámetros de la Distribución")
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Número de ensayos (n)", min_value=1, value=10, step=1)
    with col2:
        p = st.number_input("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    return n, p
 
 
def formulario_parametros_poisson():
    st.subheader("📊 Parámetros de Poisson")
    mu = st.number_input("Tasa media (μ)", min_value=0.01, value=5.0, step=0.1)
    return mu
 
 
def formulario_parametros_normales():
    st.subheader("📈 Parámetros de la Distribución Normal")
    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("Media (μ)", value=0.0, step=0.1)
    with col2:
        sigma = st.number_input("Desviación estándar (σ)", min_value=0.1, value=1.5, step=0.1)
    return mu, sigma
 
 
def formulario_parametros_exponencial():
    st.subheader("⏱️ Parámetros de la Distribución Exponencial")
    lam = st.number_input("Parámetro de tasa (λ)", min_value=0.1, value=1.0, step=0.1)
    return lam
 
 
def calcular_estadisticas_discretas(tipo: str, n: int = None, p: float = None, mu: float = None):
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
    if tipo == "normal":
        varianza = sigma ** 2
        desv = sigma
        return {"media": "μ", "varianza": varianza, "desv": desv}
    elif tipo == "exponencial":
        media = 1 / lam
        varianza = 1 / (lam ** 2)
        desv = 1 / lam
        return {"media": media, "varianza": varianza, "desv": desv}
    return {}
 
 
def modulo_validacion(tipo: str, distribucion: str, n: int = None, p: float = None, 
                      mu: float = None, sigma: float = None, lam: float = None,
                      val_x: float = None):
    st.markdown("---")
    st.markdown("### 🔍 Validación Manual de Ejercicio")
    if distribucion == "Binomial" and val_x is not None:
        comb_val = __import__('math').comb(n, int(val_x))
        prob = comb_val * (p ** val_x) * ((1 - p) ** (n - val_x))
        with st.expander("Ver cálculo detallado", expanded=True):
            st.info(
                f"**Binomial (n={n}, p={p}, x={int(val_x)}):**\n\n"
                f"1. Combinatoria $\\binom{{{n}}}{{{int(val_x)}}}$: {comb_val}\n"
                f"2. Término éxito ($p^{val_x}$): {p**val_x:.5f}\n"
                f"3. Término fracaso ($(1-p)^{n-val_x}$): {((1-p)**(n-val_x)):.5f}\n\n"
                f"👉 **P(X = {int(val_x)}) = {prob:.6f}** ({prob*100:.2f}%)"
            )
    elif distribucion == "Poisson" and val_x is not None:
        prob = (np.exp(-mu) * (mu ** val_x)) / __import__('math').factorial(int(val_x))
        with st.expander("Ver cálculo detallado", expanded=True):
            st.info(
                f"**Poisson (μ={mu}, x={int(val_x)}):**\n\n"
                f"1. Factor $e^{{-{mu}}}$: {np.exp(-mu):.5f}\n"
                f"2. Término tasa ($\\mu^{val_x}$): {mu**val_x:.5f}\n"
                f"3. Factorial ($x!$): {__import__('math').factorial(int(val_x))}\n\n"
                f"👉 **P(X = {int(val_x)}) = {prob:.6f}** ({prob*100:.2f}%)"
            )
    elif distribucion == "Normal" and val_x is not None:
        from numpy import pi, exp, sqrt
        dens = (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((val_x - mu) / sigma) ** 2)
        with st.expander("Ver resultado de validación", expanded=True):
            st.info(f"**Normal (μ={mu}, σ={sigma}, x={val_x:.2f}):**\\n\\n👉 **f(x) = {dens:.6f}**")
    elif distribucion == "Exponencial" and val_x is not None:
        prob = lam * np.exp(-lam * val_x)
        with st.expander("Ver resultado de validación", expanded=True):
            st.info(f"**Exponencial (λ={lam}, x={val_x:.2f}):**\\n\\n👉 **f(x) = {prob:.6f}**")
 
 
st.title("📊 Simulador de Variables Aleatorias")
st.markdown(
    "Universidad Nacional Abierta y a Distancia (UNAD) — Herramienta interactiva"
    " para la exploración y validación de variables aleatorias en ejercicios"
    " de probabilidad y estadística."
)
st.markdown("---")
 
tipo_variable = st.sidebar.selectbox(
    "Tipo de Variable Aleatoria",
    ["Discreta", "Continua"],
    help="Seleccione el tipo de variable para configurar el simulador.",
)
 
if tipo_variable == "Discreta":
    st.sidebar.header("⚙️ Configuración")
    distribucion = st.sidebar.radio(
        "Modelo de Distribución",
        ["Binomial", "Poisson"],
        help="Seleccione el modelo de distribución discreta.",
    )
    if distribucion == "Binomial":
        n, p = formulario_parametros_discretos()
        stats = calcular_estadisticas_discretas("binomial", n=n, p=p)
        x = np.arange(0, n + 1)
        y = _pmf_binomial(x, n, p)
        st.subheader(f"📌 Distribución Binomial: X ~ Bin(n={n}, p={p})")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Esperanza Matemática E[X]", f"{stats['media']:.4f}")
        with col2: st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
        with col3: st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")
        graficar_distribucion_discreta(x, y, f"Distribución Binomial n={n}, p={p}")
        val_x = st.number_input("Ingrese valor de x para validar:", min_value=0, max_value=int(n), value=0, step=1)
        modulo_validacion("discreta", "Binomial", n=n, p=p, val_x=val_x)
    else:
        mu = formulario_parametros_poisson()
        stats = calcular_estadisticas_discretas("poisson", mu=mu)
        x_max = int(mu * 3) + 15
        x = np.arange(0, x_max)
        y = _pmf_poisson(x, mu)
        st.subheader(f"📌 Distribución de Poisson: X ~ Pois(μ={mu})")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Esperanza Matemática E[X]", f"{stats['media']:.4f}")
        with col2: st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
        with col3: st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")
        graficar_distribucion_discreta(x, y, f"Distribución de Poisson μ={mu}")
        val_x = st.number_input("Ingrese valor de x para validar:", min_value=0, max_value=int(mu * 3), value=int(mu), step=1)
        modulo_validacion("discreta", "Poisson", mu=mu, val_x=val_x)
 
else:
    st.sidebar.header("⚙️ Configuración")
    distribucion = st.sidebar.radio(
        "Modelo de Distribución",
        ["Normal (Gaussiana)", "Exponencial"],
        help="Seleccione el modelo de distribución continua.",
    )
    if distribucion == "Normal (Gaussiana)":
        mu, sigma = formulario_parametros_normales()
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        y = _pdf_normal(x, mu, sigma)
        st.subheader(f"📌 Distribución Normal: X ~ N(μ={mu}, σ²={sigma**2})")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Esperanza Matemática E[X]", f"{mu:.4f}")
        with col2: st.metric("Varianza V[X]", f"{sigma**2:.4f}")
        with col3: st.metric("Desviación Estándar (σ)", f"{sigma:.4f}")
        graficar_distribucion_continua(x, y, f"Distribución Normal μ={mu}, σ={sigma}")
        val_x = st.number_input("Ingrese valor de x para evaluar densidad:", value=mu, step=0.1)
        modulo_validacion("continua", "Normal", sigma=sigma, val_x=val_x)
    else:
        lam = formulario_parametros_exponencial()
        x = np.linspace(0, 5 / lam, 300)
        y = _pdf_exponencial(x, lam)
        stats = calcular_estadisticas_continuas("exponencial", lam=lam)
        st.subheader(f"📌 Distribución Exponencial: X ~ Exp(λ={lam})")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Esperanza Matemática E[X]", f"{stats['media']:.4f}")
        with col2: st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
        with col3: st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")
        graficar_distribucion_continua(x, y, f"Distribución Exponencial λ={lam}")
        val_x = st.number_input("Ingrese valor de x para evaluar densidad:", value=1/lam, step=0.1)
        modulo_validacion("continua", "Exponencial", lam=lam, val_x=val_x)
