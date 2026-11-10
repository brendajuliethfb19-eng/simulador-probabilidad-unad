import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ==========================================
# CONFIGURACIÓN DE PÁGINA Y COLORES OFICIALES UNAD
# ==========================================
st.set_page_config(
    page_title="Simulador de Probabilidad - UNAD",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓",
)

# Estilos CSS con la identidad corporativa de la UNAD (Naranja y Azul Institucional)
st.markdown(
    """
<style>
    :root {
        --unad-orange: #F58220;
        --unad-blue: #000066;
        --unad-light-bg: #F8F9FA;
    }
    .main {
        background-color: #FFFFFF;
    }
    .stTitle {
        color: #000066 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stSidebar {
        background-color: #F8F9FA;
        border-right: 3px solid #F58220;
    }
    h1, h2, h3 {
        color: #000066 !important;
        font-weight: 700;
    }
    .stButton > button {
        background-color: #F58220;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #d06b15;
        color: white;
    }
</style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# FUNCIONES MATEMÁTICAS
# ==========================================
def _combinacion(n: int, k: int) -> int:
  return math.comb(n, k)


def _pmf_binomial(k: np.ndarray, n: int, p: float) -> np.ndarray:
  return np.array(
      [_combinacion(n, int(i)) * (p**i) * ((1 - p) ** (n - i)) for i in k]
  )


def _pmf_poisson(k: np.ndarray, mu: float) -> np.ndarray:
  return np.array(
      [(np.exp(-mu) * (mu**i)) / math.factorial(int(i)) for i in k]
  )


def _pdf_normal(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
  return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
      -0.5 * ((x - mu) / sigma) ** 2
  )


def _pdf_exponencial(x: np.ndarray, lam: float) -> np.ndarray:
  return lam * np.exp(-lam * x)


# ==========================================
# FUNCIONES DE GRÁFICA (Estilo UNAD)
# ==========================================
def graficar_distribucion_discreta(x: np.ndarray, y: np.ndarray, titulo: str):
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.bar(
      x,
      y,
      color="#F58220",
      alpha=0.85,
      edgecolor="#000066",
      linewidth=0.8,
      width=0.6,
  )
  ax.set_xlabel(
      "Valores de la Variable (x)",
      fontsize=10,
      fontweight="bold",
      color="#000066",
  )
  ax.set_ylabel(
      "Probabilidad P(X = x)", fontsize=10, fontweight="bold", color="#000066"
  )
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.set_title(titulo, fontsize=12, color="#000066", fontweight="bold")
  st.pyplot(fig)


def graficar_distribucion_continua(x: np.ndarray, y: np.ndarray, titulo: str):
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(x, y, color="#000066", linewidth=2.5, label="f(x)")
  ax.fill_between(x, y, color="#F58220", alpha=0.25)
  ax.set_xlabel(
      "Valores (x)", fontsize=10, fontweight="bold", color="#000066"
  )
  ax.set_ylabel(
      "Densidad f(x)", fontsize=10, fontweight="bold", color="#000066"
  )
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.legend()
  ax.set_title(titulo, fontsize=12, color="#000066", fontweight="bold")
  st.pyplot(fig)


# ==========================================
# FORMULARIOS Y CÁLCULOS
# ==========================================
def formulario_parametros_discretos():
  st.subheader("🔢 Parámetros de la Distribución")
  col1, col2 = st.columns(2)
  with col1:
    n = st.number_input("Número de ensayos (n)", min_value=1, value=10, step=1)
  with col2:
    p = st.number_input(
        "Probabilidad de éxito (p)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
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
    sigma = st.number_input(
        "Desviación estándar (σ)", min_value=0.1, value=1.5, step=0.1
    )
  return mu, sigma


def formulario_parametros_exponencial():
  st.subheader("⏱️ Parámetros de la Distribución Exponencial")
  lam = st.number_input("Parámetro de tasa (λ)", min_value=0.1, value=1.0, step=0.1)
  return lam


# ==========================================
# MÓDULO DE VALIDACIÓN MANUAL CORREGIDO
# ==========================================
def modulo_validacion(
    tipo: str,
    distribucion: str,
    n: int = None,
    p: float = None,
    mu: float = None,
    sigma: float = None,
    lam: float = None,
    val_x: float = None,
):
  st.markdown("---")
  st.markdown("### 🔍 Validación Manual de Ejercicio para Estudiantes")

  if distribucion == "Binomial" and val_x is not None:
    comb_val = math.comb(n, int(val_x))
    prob = comb_val * (p**val_x) * ((1 - p) ** (n - val_x))
    with st.expander("Ver cálculo paso a paso", expanded=True):
      st.info(
          f"**Binomial (n={n}, p={p}, x={int(val_x)}):**\n\n"
          f"1. Combinatoria $\\binom{{{n}}}{{{int(val_x)}}}$: {comb_val}\n"
          f"2. Probabilidad de éxito ($p^{val_x}$): {p**val_x:.5f}\n"
          f"3. Probabilidad de fracaso ($(1-p)^{n-val_x}$):"
          f" {((1-p)**(n-val_x)):.5f}\n\n👉 **P(X = {int(val_x)}) = {prob:.6f}**"
          f" ({prob*100:.2f}%)"
      )

  elif distribucion == "Poisson" and val_x is not None:
    prob = (np.exp(-mu) * (mu**val_x)) / math.factorial(int(val_x))
    with st.expander("Ver cálculo paso a paso", expanded=True):
      st.info(
          f"**Poisson (μ={mu}, x={int(val_x)}):**\n\n"
          f"1. Factor exponencial ($e^{{-{mu}}}$): {np.exp(-mu):.5f}\n"
          f"2. Término de tasa ($\mu^{val_x}$): {mu**val_x:.5f}\n"
          f"3. Factorial ($x!$): {math.factorial(int(val_x))}\n\n👉 **P(X ="
          f" {int(val_x)}) = {prob:.6f}** ({prob*100:.2f}%)"
      )

  elif distribucion == "Normal" and val_x is not None:
    # Corrección: Se incluyen mu y sigma correctamente en el cálculo
    dens = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((val_x - mu) / sigma) ** 2
    )
    with st.expander("Ver cálculo paso a paso", expanded=True):
      st.info(
          f"**Distribución Normal (μ={mu}, σ={sigma}, x={val_x:.2f}):**\n\n"
          f"1. Diferencia estandarizada (($x - \\mu$)/$\\sigma$):"
          f" {((val_x - mu)/sigma):.4f}\n"
          f"2. Exponencial de la campana: {np.exp(-0.5 * ((val_x - mu) / sigma) ** 2):.5f}\n\n"
          f"👉 **Valor de Densidad f(x) = {dens:.6f}**"
      )

  elif distribucion == "Exponencial" and val_x is not None:
    prob = lam * np.exp(-lam * val_x)
    with st.expander("Ver cálculo paso a paso", expanded=True):
      st.info(
          f"**Distribución Exponencial (λ={lam}, x={val_x:.2f}):**\n\n"
          f"1. Exponente ($-\\lambda x$): {-lam * val_x:.4f}\n\n👉 **Valor de"
          f" Densidad f(x) = {prob:.6f}**"
      )


# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("📊 Simulador y Validador de Variables Aleatorias")
st.markdown(
    "**Universidad Nacional Abierta y a Distancia (UNAD)** — Herramienta"
    " interactiva para la exploración pedagógica y validación manual de"
    " ejercicios estadísticos."
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
    stats = {
        "media": n * p,
        "varianza": n * p * (1 - p),
        "desv": np.sqrt(n * p * (1 - p)),
    }
    x = np.arange(0, n + 1)
    y = _pmf_binomial(x, n, p)
    st.subheader(f"📌 Distribución Binomial: X ~ Bin(n={n}, p={p})")

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Esperanza E[X]", f"{stats['media']:.4f}")
    with col2:
      st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
    with col3:
      st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")

    graficar_distribucion_discreta(x, y, f"Distribución Binomial n={n}, p={p}")
    val_x = st.number_input(
        "Ingrese un valor de x para validar a mano:",
        min_value=0,
        max_value=int(n),
        value=0,
        step=1,
    )
    modulo_validacion("discreta", "Binomial", n=n, p=p, val_x=val_x)
  else:
    mu = formulario_parametros_poisson()
    stats = {"media": mu, "varianza": mu, "desv": np.sqrt(mu)}
    x_max = int(mu * 3) + 15
    x = np.arange(0, x_max)
    y = _pmf_poisson(x, mu)
    st.subheader(f"📌 Distribución de Poisson: X ~ Pois(μ={mu})")

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Esperanza E[X]", f"{stats['media']:.4f}")
    with col2:
      st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
    with col3:
      st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")

    graficar_distribucion_discreta(x, y, f"Distribución de Poisson μ={mu}")
    val_x = st.number_input(
        "Ingrese un valor de x para validar a mano:",
        min_value=0,
        max_value=int(mu * 3),
        value=int(mu),
        step=1,
    )
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
    with col1:
      st.metric("Esperanza E[X]", f"{mu:.4f}")
    with col2:
      st.metric("Varianza V[X]", f"{sigma**2:.4f}")
    with col3:
      st.metric("Desviación Estándar (σ)", f"{sigma:.4f}")

    graficar_distribucion_continua(x, y, f"Distribución Normal μ={mu}, σ={sigma}")
    val_x = st.number_input(
        "Ingrese un valor de x para evaluar densidad:", value=mu, step=0.1
    )
    # Se pasan correctamente mu y sigma para solucionar el error
    modulo_validacion("continua", "Normal", mu=mu, sigma=sigma, val_x=val_x)
  else:
    lam = formulario_parametros_exponencial()
    x = np.linspace(0, 5 / lam, 300)
    y = _pdf_exponencial(x, lam)
    stats = {"media": 1 / lam, "varianza": 1 / (lam**2), "desv": 1 / lam}
    st.subheader(f"📌 Distribución Exponencial: X ~ Exp(λ={lam})")

    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Esperanza E[X]", f"{stats['media']:.4f}")
    with col2:
      st.metric("Varianza V[X]", f"{stats['varianza']:.4f}")
    with col3:
      st.metric("Desviación Estándar (σ)", f"{stats['desv']:.4f}")

    graficar_distribucion_continua(x, y, f"Distribución Exponencial λ={lam}")
    val_x = st.number_input(
        "Ingrese un valor de x para evaluar densidad:",
        value=float(1 / lam),
        step=0.1,
    )
    modulo_validacion("continua", "Exponencial", lam=lam, val_x=val_x)
