import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Laboratorio Manual de Probabilidad - UNAD", layout="wide"
)

st.title("🔬 Laboratorio Interactivo y Manual de Métodos Probabilísticos")
st.markdown(
    "**Universidad Nacional Abierta y a Distancia (UNAD)** — Entorno de"
    " aprendizaje práctico donde los estudiantes ingresan parámetros y valores"
    " específicos para **validar procedimientos de forma manual y en tiempo"
    " real**."
)
st.markdown("---")


# --- FUNCIONES MATEMÁTICAS ---
def comb(n, k):
  if k < 0 or k > n:
    return 0
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


# --- MENÚ LATERAL ---
st.sidebar.header("⚙️ Configuración del Experimento")
tipo_variable = st.sidebar.selectbox(
    "Tipo de Variable", ["Discreta (Binomial / Poisson)", "Continua (Normal / Exponencial)"]
)

if "Discreta" in tipo_variable:
  distribucion = st.sidebar.selectbox(
      "Seleccione el Modelo", ["Binomial", "Poisson"]
  )

  if distribucion == "Binomial":
    st.sidebar.subheader("Parámetros del Problema")
    n = st.sidebar.slider("Ensayos totales (n)", 1, 50, 10, 1)
    p = st.sidebar.slider(
        "Probabilidad de éxito (p)", 0.01, 1.0, 0.5, 0.01
    )

    x = np.arange(0, n + 1)
    y = binom_pmf(x, n, p)

    st.subheader(f"📌 Modelo Binomial: X ~ Bin(n={n}, p={p})")

    # ZONA DE INTERACCIÓN MANUAL PARA EL ESTUDIANTE
    st.markdown("### 🧮 Módulo de Práctica y Validación Manual")
    st.markdown(
        "Digite el valor exacto que desea evaluar para comprobar su cálculo"
        " manual:"
    )

    col_1, col_2 = st.columns([1, 1])
    with col_1:
      k_user = st.number_input(
          "Valor de éxito a probar (x):",
          min_value=0,
          max_value=n,
          value=int(n / 2),
          step=1,
      )

    with col_2:
      # Cálculos manuales detallados
      c_val = comb(n, k_user)
      term_exito = p**k_user
      term_fracaso = (1 - p) ** (n - k_user)
      prob_final = c_val * term_exito * term_fracaso

      st.info(
          f"**Desarrollo Paso a Paso para x = {k_user}:**\n"
          f"1. Combinatoria $\\binom{{{n}}}{{{k_user}}} = {c_val}$\n"
          f"2. Potencia de éxito ($p^{k_user}$) = {term_exito:.5f}\n"
          f"3. Potencia de fracaso ($(1-p)^{{n-k_user}}$) = {term_fracaso:.5f}\n\n"
          f" 👉 **Probabilidad P(X = {k_user}) = {prob_final:.5f}**"
          f" (**{prob_final*100:.2f}%**)"
      )

  else:  # Poisson
    st.sidebar.subheader("Parámetros del Problema")
    mu = st.sidebar.slider(
        "Tasa media de ocurrencia (mu)", 0.1, 20.0, 4.0, 0.5
    )

    x = np.arange(0, int(mu * 3) + 10)
    y = poisson_pmf(x, mu)

    st.subheader(f"📌 Modelo de Poisson: X ~ Pois(mu={mu})")

    st.markdown("### 🧮 Módulo de Práctica y Validación Manual")
    col_1, col_2 = st.columns([1, 1])
    with col_1:
      k_user = st.number_input(
          "Valor de ocurrencia a probar (x):",
          min_value=0,
          max_value=int(max(x)),
          value=int(mu),
          step=1,
      )

    with col_2:
      prob_final = (np.exp(-mu) * (mu**k_user)) / math.factorial(k_user)
      st.info(
          f"**Desarrollo Paso a Paso para x = {k_user}:**\n"
          f"1. Constante exponencial ($e^{{-{mu}}}$) = {np.exp(-mu):.5f}\n"
          f"2. Tasa elevada ($\mu^{k_user}$) = {mu**k_user:.5f}\n"
          f"3. Factorial ($x!$) = {math.factorial(k_user)}\n\n"
          f" 👉 **Probabilidad P(X = {k_user}) = {prob_final:.5f}**"
          f" (**{prob_final*100:.2f}%**)"
      )

  # Gráfico visual dinámico
  st.markdown("### 📈 Visualización Gráfica de la Distribución")
  fig, ax = plt.subplots(figsize=(10, 3.5))
  ax.bar(x, y, color="#1f77b4", alpha=0.85, edgecolor="black", width=0.7)
  ax.set_xlabel("Valores de x", fontweight="bold")
  ax.set_ylabel("Probabilidad P(X = x)", fontweight="bold")
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  st.pyplot(fig)

else:  # Continuas
  distribucion = st.sidebar.selectbox(
      "Seleccione el Modelo", ["Normal (Gaussiana)", "Exponencial"]
  )

  if distribucion == "Normal (Gaussiana)":
    st.sidebar.subheader("Parámetros del Problema")
    mu = st.sidebar.slider("Media (mu)", -10.0, 10.0, 0.0, 0.5)
    sigma = st.sidebar.slider("Desviación Estándar (sigma)", 0.1, 5.0, 1.0, 0.1)

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    y = norm_pdf(x, mu, sigma)

    st.subheader(f"📌 Modelo Normal: X ~ N(mu={mu}, sigma^2={sigma**2})")

    st.markdown("### 🧮 Módulo de Práctica y Validación Manual")
    col_1, col_2 = st.columns([1, 1])
    with col_1:
      x_user = st.number_input(
          "Introduce el valor de x a evaluar:",
          min_value=float(x[0]),
          max_value=float(x[-1]),
          value=float(mu),
          step=0.1,
      )
    with col_2:
      densidad = norm_pdf(x_user, mu, sigma)
      st.info(
          f"**Evaluación Manual de Densidad:**\nPara x = {x_user:.2f} en la"
          f" normal de media {mu} y sigma {sigma}.\n\n 👉 **Valor f(x) ="
          f" {densidad:.5f}**"
      )

  else:
    st.sidebar.subheader("Parámetros del Problema")
    lam = st.sidebar.slider("Parámetro de tasa (lambda)", 0.1, 5.0, 1.0, 0.1)

    x = np.linspace(0, 5 / lam, 300)
    y = expon_pdf(x, lam)

    st.subheader(f"📌 Modelo Exponencial: X ~ Exp(lambda={lam})")

    st.markdown("### 🧮 Módulo de Práctica y Validación Manual")
    col_1, col_2 = st.columns([1, 1])
    with col_1:
      x_user = st.number_input(
          "Introduce el valor de x a evaluar:",
          min_value=float(x[0]),
          max_value=float(x[-1]),
          value=float(1.0 / lam),
          step=0.1,
      )
    with col_2:
      densidad = expon_pdf(x_user, lam)
      st.info(
          f"**Evaluación Manual de Densidad:**\nPara x = {x_user:.2f} con tasa"
          f" lambda = {lam}.\n\n 👉 **Valor f(x) = {densidad:.5f}**"
      )

  st.markdown("### 📈 Curva de Densidad de Probabilidad")
  fig, ax = plt.subplots(figsize=(10, 3.5))
  ax.plot(x, y, color="#2ca02c", linewidth=2.5)
  ax.fill_between(x, y, color="#2ca02c", alpha=0.2)
  ax.set_xlabel("Valores de x", fontweight="bold")
  ax.set_ylabel("Densidad f(x)", fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5)
  st.pyplot(fig)