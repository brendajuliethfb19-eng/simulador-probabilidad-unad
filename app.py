import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Simulador y Laboratorio Probabilístico - UNAD",
    layout="wide",
)

st.title("🚀 Laboratorio Interactivo Avanzado de Métodos Probabilísticos")
st.markdown(
    "**Universidad Nacional Abierta y a Distancia (UNAD)** — Entorno virtual de"
    " aprendizaje dinámico para la simulación y **validación manual de"
    " ejercicios**."
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


# --- PANEL LATERAL ---
st.sidebar.header("🎛️ Panel de Control")
tipo_variable = st.sidebar.selectbox(
    "Tipo de Variable Aleatoria", ["Discreta", "Continua"]
)

if tipo_variable == "Discreta":
  distribucion = st.sidebar.selectbox(
      "Modelo de Distribución", ["Binomial", "Poisson"]
  )

  if distribucion == "Binomial":
    st.sidebar.subheader("Parámetros Binomiales")
    n = st.sidebar.slider("Ensayos (n)", 1, 50, 20, 1)
    p = st.sidebar.slider("Probabilidad (p)", 0.01, 1.0, 0.5, 0.01)

    x = np.arange(0, n + 1)
    y = binom_pmf(x, n, p)
    media = n * p
    varianza = n * p * (1 - p)
    desv = np.sqrt(varianza)

    st.subheader("📌 Modelo Binomial Activo")

  else:
    st.sidebar.subheader("Parámetros de Poisson")
    mu = st.sidebar.slider("Tasa media (mu)", 0.1, 30.0, 5.0, 0.5)

    x = np.arange(0, int(mu * 3) + 15)
    y = poisson_pmf(x, mu)
    media = mu
    varianza = mu
    desv = np.sqrt(mu)

    st.subheader("📌 Modelo de Poisson Activo")

  # Métricas
  c1, c2, c3 = st.columns(3)
  c1.metric("Esperanza E[X]", f"{media:.4f}")
  c2.metric("Varianza V[X]", f"{varianza:.4f}")
  c3.metric("Desviación Estándar", f"{desv:.4f}")

  # Pestañas
  tab1, tab2 = st.tabs(
      ["📈 Visualización Gráfica", "🔍 Validador y Calculadora Manual"]
  )

  with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y, color="#1f77b4", alpha=0.85, edgecolor="black", width=0.7)
    ax.set_xlabel("Valores (x)", fontweight="bold")
    ax.set_ylabel("Probabilidad P(X = x)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)

  with tab2:
    st.markdown("### 👨‍💻 Espacio de Validación para Estudiantes")
    val_ingresado = st.number_input(
        "Introduce el valor de x a evaluar:",
        int(min(x)),
        int(max(x)),
        int(media),
        1,
    )

    if distribucion == "Binomial":
      c_val = comb(n, val_ingresado)
      prob = c_val * (p**val_ingresado) * ((1 - p) ** (n - val_ingresado))
      st.info(
          f"**Resultado de Validación para Binomial (n={n}, p={p}, x={val_ingresado}):**"
          f" \n- Coeficiente combinatorio: {c_val} \n- **Probabilidad P(X ="
          f" {val_ingresado}) = {prob:.6f}** ({prob*100:.2f}%)"
      )
    else:
      prob = (np.exp(-mu) * (mu**val_ingresado)) / math.factorial(val_ingresado)
      st.info(
          f"**Resultado de Validación para Poisson (mu={mu}, x={val_ingresado}):**"
          f" \n- **Probabilidad P(X = {val_ingresado}) = {prob:.6f}**"
          f" ({prob*100:.2f}%)"
      )

else:
  distribucion = st.sidebar.selectbox(
      "Modelo de Distribución", ["Normal (Gaussiana)", "Exponencial"]
  )

  if distribucion == "Normal (Gaussiana)":
    st.sidebar.subheader("Parámetros Normales")
    mu = st.sidebar.slider("Media (mu)", -15.0, 15.0, 0.0, 0.5)
    sigma = st.sidebar.slider("Desviación (sigma)", 0.1, 6.0, 1.5, 0.1)

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    y = norm_pdf(x, mu, sigma)
    media = mu
    varianza = sigma**2
    desv = sigma
    st.subheader("📌 Modelo Normal Activo")

  else:
    st.sidebar.subheader("Parámetros Exponenciales")
    lam = st.sidebar.slider("Tasa (lambda)", 0.1, 5.0, 1.0, 0.1)

    x = np.linspace(0, 5 / lam, 300)
    y = expon_pdf(x, lam)
    media = 1 / lam
    varianza = 1 / (lam**2)
    desv = 1 / lam
    st.subheader("📌 Modelo Exponencial Activo")

  c1, c2, c3 = st.columns(3)
  c1.metric("Esperanza E[X]", f"{media:.4f}")
  c2.metric("Varianza V[X]", f"{varianza:.4f}")
  c3.metric("Desviación Estándar", f"{desv:.4f}")

  tab1, tab2 = st.tabs(
      ["📈 Curva de Densidad", "🔍 Validador y Calculadora Manual"]
  )

  with tab1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="#2ca02c", linewidth=2.5)
    ax.fill_between(x, y, color="#2ca02c", alpha=0.2)
    ax.set_xlabel("Valores (x)", fontweight="bold")
    ax.set_ylabel("Densidad f(x)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

  with tab2:
    st.markdown("### 👨‍💻 Espacio de Validación para Estudiantes")
    val_c = st.number_input("Introduce un valor de x:", float(x[0]), float(x[-1]), float(media), 0.1)

    if distribucion == "Normal (Gaussiana)":
      dens = norm_pdf(val_c, mu, sigma)
      st.info(f"**Densidad Normal en x = {val_c:.2f}:** f(x) = {dens:.6f}")
    else:
      dens = expon_pdf(val_c, lam)
      st.info(f"**Densidad Exponencial en x = {val_c:.2f}:** f(x) = {dens:.6f}")