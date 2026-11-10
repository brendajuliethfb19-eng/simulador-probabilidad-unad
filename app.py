import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página web
st.set_page_config(
    page_title="Simulador de Probabilidad - UNAD",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Simulador Interactivo y Validador de Variables Aleatorias")
st.markdown(
    "**Proyecto Académico de Alto Impacto** — Universidad Nacional Abierta y a"
    " Distancia (UNAD). Herramienta diseñada para la exploración visual y la"
    " **validación manual de ejercicios de probabilidad**."
)
st.markdown("---")


# --- FUNCIONES MATEMÁTICAS ---
def comb(n, k):
  return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def binom_pmf(k, n, p):
  return [comb(n, int(i)) * (p**i) * ((1 - p) ** (n - i)) for i in k]


def poisson_pmf(k, mu):
  return [(np.exp(-mu) * (mu**i)) / math.factorial(int(i)) for i in k]


def norm_pdf(x, mu, sigma):
  return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
      -0.5 * ((x - mu) / sigma) ** 2
  )


def expon_pdf(x, lam):
  return lam * np.exp(-lam * x)


# Menú lateral de navegación
st.sidebar.header("⚙️ Panel de Control")
tipo_variable = st.sidebar.selectbox(
    "Tipo de Variable Aleatoria", ["Discreta", "Continua"]
)

if tipo_variable == "Discreta":
  distribucion = st.sidebar.selectbox(
      "Modelo de Distribución", ["Binomial", "Poisson"]
  )

  if distribucion == "Binomial":
    st.sidebar.subheader("Parámetros Binomiales")
    n = st.sidebar.slider(
        "Número de ensayos (n)", min_value=1, max_value=50, value=20, step=1
    )
    p = st.sidebar.slider(
        "Probabilidad de éxito (p)", min_value=0.01, max_value=1.0, value=0.5, step=0.01
    )

    x = np.arange(0, n + 1)
    y = binom_pmf(x, n, p)
    media = n * p
    varianza = n * p * (1 - p)
    desv = np.sqrt(varianza)

    st.subheader(f"📌 Distribución Binomial: X ~ Bin(n={n}, p={p})")

  elif distribucion == "Poisson":
    st.sidebar.subheader("Parámetros de Poisson")
    mu = st.sidebar.slider(
        "Tasa media de ocurrencia (mu)",
        min_value=0.1,
        max_value=30.0,
        value=5.0,
        step=0.5,
    )

    x = np.arange(0, int(mu * 3) + 15)
    y = poisson_pmf(x, mu)
    media = mu
    varianza = mu
    desv = np.sqrt(mu)

    st.subheader(f"📌 Distribución de Poisson: X ~ Pois(mu={mu})")

  # Métricas clave
  col1, col2, col3 = st.columns(3)
  col1.metric("Esperanza Matemática E[X]", f"{media:.4f}")
  col2.metric("Varianza V[X]", f"{varianza:.4f}")
  col3.metric("Desviación Estándar (sigma)", f"{desv:.4f}")

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

  # --- SECCIÓN DE ALTO IMPACTO: VALIDACIÓN MANUAL DE EJERCICIOS ---
  st.markdown("---")
  st.markdown(
      "### 🔍 Módulo de Validación Manual para Estudiantes"
  )
  st.markdown(
      "Ingresa un valor específico de $x$ para validar el cálculo paso a paso y"
      " compararlo con tu procedimiento manual:"
  )

  val_ingresado = st.number_input(
      "Introduce el valor de x a evaluar:",
      min_value=int(min(x)),
      max_value=int(max(x)),
      value=int(media),
      step=1,
  )

  if distribucion == "Binomial":
    # Cálculo detallado Binomial
    comb_val = comb(n, val_ingresado)
    prob_exacta = comb_val * (p**val_ingresado) * ((1 - p) ** (n - val_ingresado))

    st.info(
        f"**Validación Paso a Paso para Binomial (n={n}, p={p}, x={val_ingresado}):**"
        f" \n1. **Combinatoria $\\binom{{{n}}}{{{val_ingresado}}}$:**"
        f" {comb_val} \n2. **Término de éxito ($p^{val_ingresado}$):**"
        f" {p**val_ingresado:.5f} \n3. **Término de fracaso"
        f" ($(1-p)^{n-val_ingresado}$):**"
        f" {((1-p)**(n-val_ingresado)):.5f} \n\n 👉 **Resultado P(X ="
        f" {val_ingresado}) = {prob_exacta:.6f}** (Equivalente a"
        f" **{prob_exacta*100:.2f}%**)"
    )
  else:
    # Cálculo detallado Poisson
    prob_exacta = (np.exp(-mu) * (mu**val_ingresado)) / math.factorial(
        val_ingresado
    )
    st.info(
        f"**Validación Paso a Paso para Poisson (mu={mu}, x={val_ingresado}):**"
        f" \n1. **Factor e ajustado ($e^{{-{mu}}}$):**"
        f" {np.exp(-mu):.5f} \n2. **Término de tasa ($\mu^{val_ingresado}$):**"
        f" {mu**val_ingresado:.5f} \n3. **Factorial ($x!$):**"
        f" {math.factorial(val_ingresado)} \n\n 👉 **Resultado P(X ="
        f" {val_ingresado}) = {prob_exacta:.6f}** (Equivalente a"
        f" **{prob_exacta*100:.2f}%**)"
    )

else:
  distribucion = st.sidebar.selectbox(
      "Modelo de Distribución", ["Normal (Gaussiana)", "Exponencial"]
  )

  if distribucion == "Normal (Gaussiana)":
    st.sidebar.subheader("Parámetros Normales")
    mu = st.sidebar.slider(
        "Media (mu)", min_value=-15.0, max_value=15.0, value=0.0, step=0.5
    )
    sigma = st.sidebar.slider(
        "Desviación Estándar (sigma)",
        min_value=0.1,
        max_value=6.0,
        value=1.5,
        step=0.1,
    )

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    y = norm_pdf(x, mu, sigma)
    media = mu
    varianza = sigma**2
    desv = sigma

    st.subheader(f"📌 Distribución Normal: X ~ N(mu={mu}, sigma^2={sigma**2})")

  elif distribucion == "Exponencial":
    st.sidebar.subheader("Parámetros Exponenciales")
    lam = st.sidebar.slider(
        "Parámetro de tasa (lambda)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )

    x = np.linspace(0, 5 / lam, 300)
    y = expon_pdf(x, lam)
    media = 1 / lam
    varianza = 1 / (lam**2)
    desv = 1 / lam

    st.subheader(f"📌 Distribución Exponencial: X ~ Exp(lambda={lam})")

  col1, col2, col3 = st.columns(3)
  col1.metric("Esperanza Matemática E[X]", f"{media:.4f}")
  col2.metric("Varianza V[X]", f"{varianza:.4f}")
  col3.metric("Desviación Estándar (sigma)", f"{desv:.4f}")

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

  st.markdown("---")
  st.markdown(
      "### 🔍 Módulo de Validación Manual para Estudiantes"
  )
  val_ingresado_cont = st.number_input(
      "Introduce un valor de x para evaluar la función de densidad:",
      min_value=float(x[0]),
      max_value=float(x[-1]),
      value=float(media),
      step=0.1,
  )

  if distribucion == "Normal (Gaussiana)":
    densidad_val = norm_pdf(val_ingresado_cont, mu, sigma)
    st.info(
        f"**Validación Manual - Normal (mu={mu}, sigma={sigma}):**\nEvaluando"
        f" $x = {val_ingresado_cont:.2f}$ en la función de densidad"
        f" gaussiana.\n\n 👉 **Valor de Densidad f(x) ="
        f" {densidad_val:.6f}**"
    )
  else:
    densidad_val = expon_pdf(val_ingresado_cont, lam)
    st.info(
        f"**Validación Manual - Exponencial (lambda={lam}):**\nEvaluando"
        f" $x = {val_ingresado_cont:.2f}$ en la función exponencial.\n\n 👉"
        f" **Valor de Densidad f(x) = {densidad_val:.6f}**"
    )