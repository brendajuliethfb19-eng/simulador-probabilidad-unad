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

st.title("📊 Simulador Interactivo de Variables Aleatorias y Probabilidad")
st.markdown(
    "**Curso de Métodos Probabilísticos** — Universidad Nacional Abierta y a"
    " Distancia (UNAD). Modifica los parámetros en el panel izquierdo para"
    " observar el comportamiento dinámico de los modelos."
)
st.markdown("---")


# --- FUNCIONES MATEMÁTICAS PURAS (Sin bloqueos de SciPy) ---
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

  # Mostrar métricas clave
  col1, col2, col3 = st.columns(3)
  col1.metric("Esperanza Matemática E[X]", f"{media:.4f}")
  col2.metric("Varianza V[X]", f"{varianza:.4f}")
  col3.metric("Desviación Estándar (sigma)", f"{desv:.4f}")

  st.markdown("### 📈 Función de Masa de Probabilidad (PMF)")

  # Gráfico profesional con Matplotlib
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.bar(
      x,
      y,
      color="#1f77b4",
      alpha=0.85,
      edgecolor="black",
      linewidth=0.6,
      width=0.7,
  )
  ax.set_xlabel("Valores de la Variable (x)", fontsize=11, fontweight="bold")
  ax.set_ylabel("Probabilidad P(X = x)", fontsize=11, fontweight="bold")
  ax.grid(axis="y", linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  st.pyplot(fig)

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

  # Mostrar métricas clave
  col1, col2, col3 = st.columns(3)
  col1.metric("Esperanza Matemática E[X]", f"{media:.4f}")
  col2.metric("Varianza V[X]", f"{varianza:.4f}")
  col3.metric("Desviación Estándar (sigma)", f"{desv:.4f}")

  st.markdown("### 📈 Función de Densidad de Probabilidad (PDF)")

  # Gráfico de línea continuo con área sombreada profesional
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.plot(x, y, color="#2ca02c", linewidth=2.5, label="Densidad f(x)")
  ax.fill_between(x, y, color="#2ca02c", alpha=0.2)
  ax.set_xlabel("Valores (x)", fontsize=11, fontweight="bold")
  ax.set_ylabel("Densidad f(x)", fontsize=11, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.legend()
  st.pyplot(fig)