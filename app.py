"""Simulador Interactivo de Variables Aleatorias - UNAD"""
import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import List, Tuple
 
 
# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Simulador de Variables Aleatorias - UNAD",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)
 
 
# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown(
    """
<style>
    .main {
        padding: 2rem 1rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
    }
    .stSlider > div > div > div > div {
        background-color: #1f77b4;
    }
    h1, h2, h3 {
        color: #2c3e50 !important;
    }
    .metric-card {
        background: #fff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }
    .stExpander > summary {
        font-weight: 600;
        color: #2c3e50;
    }
</style>
    """,
    unsafe_allow_html=True,
)
 
 
# --- FUNCIONES AUXILIARIES ---
def _combinacion(n: int, k: int) -> int:
    """Calcula el binomio C(n, k)."""
    return math.comb(n, k)
 
 
def _pmf_binomial(k: np.ndarray, n: int, p: float) -> np.ndarray:
    """PMF de la distribución binomial."""
    return np.array([_combinacion(n, int(i)) * (p ** i) * ((1 - p) ** (n - i)) for i in k])
 
 
def _pmf_poisson(k: np.ndarray, mu: float) -> np.ndarray:
    """PMF de la distribución de Poisson."""
    return np.array([(np.exp(-mu) * (mu ** i)) / math.factorial(int(i)) for i in k])
 
 
def _pdf_normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """PDF de la distribución normal."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
 
 
def _pdf_exponencial(x: np.ndarray, lam: float) -> np.ndarray:
    """PDF de la distribución exponencial."""
    return lam * np.exp(-lam * x)
 
 
# --- HELPER PARA FORMATO DE RESULTADOS ---
def formatear_probabilidad(val: float) -> str:
    """Formatea un valor de probabilidad con porcentaje y notación científica."""
    return f"{val:.6f} ({val*100:.2f}%)"
 
 
# --- INTERFAZ PRINCIPAL ---
st.title("📊 Simulador Interactivo de Variables Aleatorias")
st.markdown(
    "**Universidad Nacional Abierta y a Distancia (UNAD)** — Herramienta diseñada"
    " para la exploración visual y la validación de ejercicios de probabilidad."
)
st.markdown("---")
 
 
# --- MENÚ LATERAL ---
st.sidebar.header("⚙️ Panel de Control")
modo = st.sidebar.radio(
    "Modo de Trabajo",
    ["Simulador", "Validador de Ejercicios"],
    help="Seleccione el modo de trabajo deseado.",
)
 
 
# --- LÓGICA SEGÚN TIPO DE VARIABLE ---
def _render_distribucion_discreta():
    """Renderiza distribuciones discretas (Binomial y Poisson)."""
    distribucion = st.sidebar.selectbox(
        "Modelo de Distribución",
        ["Binomial", "Poisson"],
        key="distribucion_discreta",
    )
 
    if distribucion == "Binomial":
        st.sidebar.subheader("Parámetros Binomiales")
        n = st.sidebar.slider(
            "Número de ensayos (n)",
            min_value=1,
            max_value=50,
            value=20,
            step=1,
            help="Número total de ensayos independientes.",
        )
        p = st.sidebar.slider(
            "Probabilidad de éxito (p)",
            min_value=0.01,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Probabilidad de éxito en cada ensayo.",
        )
 
        x = np.arange(0, n + 1)
        y = _pmf_binomial(x, n, p)
        media = n * p
        varianza = n * p * (1 - p)
        desv = np.sqrt(varianza)
 
        st.subheader(f"📌 Distribución Binomial: X ~ Bin(n={n}, p={p})")
 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{media:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{varianza:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{desv:.4f}")
 
        st.markdown("### 📈 Gráfico de Distribución")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(
            x,
            y,
            color="#1f77b4",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            width=0.7,
        )
        ax.set_xlabel("Valores de la Variable (x)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probabilidad P(X = x)", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
 
        # --- Módulo de Validación Manual ---
        st.markdown("---")
        st.markdown("### 🔍 Validador de Ejercicios Estudiantiles")
        st.markdown(
            "Ingresa un valor específico de $x$ para validar el cálculo paso a paso"
            " y compararlo con tu procedimiento manual:"
        )
 
        val_ingresado = st.number_input(
            "Introduce el valor de x a evaluar:",
            min_value=int(min(x)),
            max_value=int(max(x)),
            value=int(media),
            step=1,
            help="Selecciona un valor dentro del rango de la distribución.",
        )
 
        comb_val = _combinacion(n, val_ingresado)
        prob_exacta = comb_val * (p ** val_ingresado) * ((1 - p) ** (n - val_ingresado))
 
        with st.expander("Ver cálculo detallado", expanded=True):
            st.info(
                f"**Validación Paso a Paso para Binomial (n={n}, p={p}, x={val_ingresado}):**"
                f"\n\n1. **Combinatoria $\\binom{{{n}}}{{{val_ingresado}}}$:**"
                f" {comb_val}"
                f"\n2. **Término de éxito ($p^{val_ingresado}$):**"
                f" {p**val_ingresado:.5f}"
                f"\n3. **Término de fracaso ($(1-p)^{n-val_ingresado}$):**"
                f" {((1-p)**(n-val_ingresado)):.5f}"
                f"\n\n👉 **Resultado:** P(X = {val_ingresado}) = {prob_exacta:.6f}"
                f" (Equivalente a **{prob_exacta*100:.2f}%**)"
            )
 
    elif distribucion == "Poisson":
        st.sidebar.subheader("Parámetros de Poisson")
        mu = st.sidebar.slider(
            "Tasa media de ocurrencia (μ)",
            min_value=0.1,
            max_value=30.0,
            value=5.0,
            step=0.5,
            help="Número promedio de eventos en un intervalo dado.",
        )
 
        x_max = int(mu * 3) + 15
        x = np.arange(0, x_max)
        y = _pmf_poisson(x, mu)
        media = mu
        varianza = mu
        desv = np.sqrt(mu)
 
        st.subheader(f"📌 Distribución de Poisson: X ~ Pois(mu={mu})")
 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{media:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{varianza:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{desv:.4f}")
 
        st.markdown("### 📈 Gráfico de Distribución")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.bar(
            x,
            y,
            color="#ff7f0e",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            width=0.7,
        )
        ax.set_xlabel("Valores de la Variable (x)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Probabilidad P(X = x)", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
 
        # --- Módulo de Validación Manual ---
        st.markdown("---")
        st.markdown("### 🔍 Validador de Ejercicios Estudiantiles")
        st.markdown(
            "Ingresa un valor específico de $x$ para validar el cálculo paso a paso"
            " y compararlo con tu procedimiento manual:"
        )
 
        val_ingresado = st.number_input(
            "Introduce el valor de x a evaluar:",
            min_value=int(min(x)),
            max_value=int(max(x)),
            value=int(media),
            step=1,
            help="Selecciona un valor dentro del rango de la distribución.",
        )
 
        prob_exacta = (np.exp(-mu) * (mu ** val_ingresado)) / math.factorial(val_ingresado)
 
        with st.expander("Ver cálculo detallado", expanded=True):
            st.info(
                f"**Validación Paso a Paso para Poisson (μ={mu}, x={val_ingresado}):**"
                f"\n\n1. **Factor $e^{{-{mu}}}$:**"
                f" {np.exp(-mu):.5f}"
                f"\n2. **Término de tasa ($\\mu^{val_ingresado}$):**"
                f" {mu**val_ingresado:.5f}"
                f"\n3. **Factorial ($x!$):**"
                f" {math.factorial(val_ingresado)}"
                f"\n\n👉 **Resultado:** P(X = {val_ingresado}) = {prob_exacta:.6f}"
                f" (Equivalente a **{prob_exacta*100:.2f}%**)"
            )
 
 
def _render_distribucion_continua():
    """Renderiza distribuciones continuas (Normal y Exponencial)."""
    distribucion = st.sidebar.selectbox(
        "Modelo de Distribución",
        ["Normal (Gaussiana)", "Exponencial"],
        key="distribucion_continua",
    )
 
    if distribucion == "Normal (Gaussiana)":
        st.sidebar.subheader("Parámetros Normales")
        mu = st.sidebar.slider(
            "Media (μ)",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
        )
        sigma = st.sidebar.slider(
            "Desviación Estándar (σ)",
            min_value=0.1,
            max_value=6.0,
            value=1.5,
            step=0.1,
        )
 
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        y = _pdf_normal(x, mu, sigma)
        media = mu
        varianza = sigma ** 2
        desv = sigma
 
        st.subheader(f"📌 Distribución Normal: X ~ N(mu={mu}, sigma²={sigma**2})")
 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{media:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{varianza:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{desv:.4f}")
 
        st.markdown("### 📈 Curva de Densidad")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(x, y, color="#2ca02c", linewidth=2.5, label="Densidad f(x)")
        ax.fill_between(x, y, color="#2ca02c", alpha=0.2)
        ax.set_xlabel("Valores (x)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Densidad f(x)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        st.pyplot(fig)
 
        # --- Módulo de Validación Manual ---
        st.markdown("---")
        st.markdown("### 🔍 Validador de Ejercicios Estudiantiles")
        val_ingresado_cont = st.number_input(
            "Introduce un valor de x para evaluar la función de densidad:",
            min_value=float(x[0]),
            max_value=float(x[-1]),
            value=float(media),
            step=0.1,
            help="Selecciona un valor dentro del rango de la distribución.",
        )
 
        densidad_val = _pdf_normal(val_ingresado_cont, mu, sigma)
 
        with st.expander("Ver resultado de validación"):
            st.info(
                f"**Validación Manual - Normal (μ={mu}, σ={sigma}):**"
                f"\n\nEvaluando $x = {val_ingresado_cont:.2f}$ en la función de densidad"
                f" gaussiana."
                f"\n\n👉 **Valor de Densidad f(x) = {densidad_val:.6f}**"
            )
 
    elif distribucion == "Exponencial":
        st.sidebar.subheader("Parámetros Exponenciales")
        lam = st.sidebar.slider(
            "Parámetro de tasa (λ)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )
 
        x = np.linspace(0, 5 / lam, 300)
        y = _pdf_exponencial(x, lam)
        media = 1 / lam
        varianza = 1 / (lam ** 2)
        desv = 1 / lam
 
        st.subheader(f"📌 Distribución Exponencial: X ~ Exp(lambda={lam})")
 
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Esperanza Matemática E[X]", f"{media:.4f}")
        with col2:
            st.metric("Varianza V[X]", f"{varianza:.4f}")
        with col3:
            st.metric("Desviación Estándar (σ)", f"{desv:.4f}")
 
        st.markdown("### 📈 Curva de Densidad")
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(x, y, color="#d62728", linewidth=2.5, label="Densidad f(x)")
        ax.fill_between(x, y, color="#d62728", alpha=0.2)
        ax.set_xlabel("Valores (x)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Densidad f(x)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()
        st.pyplot(fig)
 
        # --- Módulo de Validación Manual ---
        st.markdown("---")
        st.markdown("### 🔍 Validador de Ejercicios Estudiantiles")
        val_ingresado_cont = st.number_input(
            "Introduce un valor de x para evaluar la función de densidad:",
            min_value=float(x[0]),
            max_value=float(x[-1]),
            value=float(media),
            step=0.1,
            help="Selecciona un valor dentro del rango de la distribución.",
        )
 
        densidad_val = _pdf_exponencial(val_ingresado_cont, lam)
 
        with st.expander("Ver resultado de validación"):
            st.info(
                f"**Validación Manual - Exponencial (λ={lam}):**"
                f"\n\nEvaluando $x = {val_ingresado_cont:.2f}$ en la función exponencial."
                f"\n\n👉 **Valor de Densidad f(x) = {densidad_val:.6f}**"
            )
 
 
# --- EJECUCIÓN PRINCIPAL ---
if modo == "Simulador":
    tipo_variable = st.sidebar.selectbox(
        "Tipo de Variable Aleatoria",
        ["Discreta", "Continua"],
        help="Seleccione el tipo de variable para continuar.",
    )
 
    if tipo_variable == "Discreta":
        _render_distribucion_discreta()
    else:
        _render_distribucion_continua()
 
elif modo == "Validador de Ejercicios":
    st.sidebar.header("📋 Modo Validador")
    tipo_variable = st.sidebar.selectbox(
        "Tipo de Variable Aleatoria",
        ["Discreta", "Continua"],
        help="Seleccione el tipo de variable a validar.",
    )
 
    if tipo_variable == "Discreta":
        _render_distribucion_discreta()
    else:
        _render_distribucion_continua()
 
st.markdown("---")
st.caption("© 2026 Simulador UNAD - Para fines educativos y académicos")