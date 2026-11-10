import math
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Configuración general de la página en modo ancho (wide)
st.set_page_config(
    page_title="Simulador y Laboratorio Probabilístico - UNAD",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo visual y encabezado principal
st.title("🚀 Laboratorio Interactivo Avanzado de Métodos Probabilísticos")
st.markdown(
    "**Universidad Nacional Abierta y a Distancia (UNAD)** — Entorno virtual de"
    " aprendizaje dinámico para la simulación, análisis y **validación manual de"
    " modelos probabilísticos**."
)
st.markdown("---")

# --- FUNCIONES MATEMÁTICAS NATIVAS ---


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


# --- PANEL LATERAL DE CONFIGURACIÓN ---
st.sidebar.header("🎛️ Panel de Control Académico")
tipo_variable = st.sidebar.selectbox(
    "1. Tipo de Variable Aleatoria", ["Discreta", "Continua"]
)

if tipo_variable == "Discreta":
  distribucion = st.sidebar.selectbox(
      "2. Seleccione el Modelo", ["Binomial", "Poisson"]
  )

  if distribucion == "Binomial":
    st.sidebar.subheader("Parámetros Binomiales")
    n = st.sidebar.slider(
        "Número total de ensayos (n)", min_value=1, max_value=80, value=25, step=1
    )
    p = st.sidebar.slider(
        "Probabilidad de éxito en cada ensayo (p)",
        min_value=0.01,
        max_value=1.0,
        value=0.45,
        step=0.01,
    )

    x = np.arange(0, n + 1)
    y = binom_pmf(x, n, p)
    media = n * p
    varianza = n * p * (1 - p)
    desv = np.sqrt(varianza)

    st.subheader(
        f"📌 Modelo Binomial Activo: $X \\sim \\text{{Bin}}(n={n}, p={p})$"
    )

  elif distribucion == "Poisson":
    st.sidebar.subheader("Parámetros de Poisson")
    mu = st.sidebar.slider(
        "Tasa media de ocurrencia en el intervalo ($\mu$)",
        min_value=0.1,
        max_value=40.0,
        value=6.0,
        step=0.5,
    )

    x = np.arange(0, int(mu * 3) + 20)
    y = poisson_pmf(x, mu)
    media = mu
    varianza = mu
    desv = np.sqrt(mu)

    st.subheader(
        f"📌 Modelo de Poisson Activo: $X \\sim \\text{{Pois}}(\mu={mu})$"
    )

  # Métricas estadísticas ejecutivas
  col1, col2, col3 = st.columns(3)
  col1.metric("Esperanza Matemática $E[X]$", f"{media:.4f}")
  col2.metric("Varianza Teórica $V[X]$", f"{varianza:.4f}")
  col3.metric("Desviación Estándar $\\sigma$", f"{desv:.4f}")

  # Pestañas principales para organizar la interacción del estudiante
  pestana_grafico, pestana_validacion = st.tabs(
      ["📈 Visualización Gráfica y Comportamiento", "🔍 Validador y Calculadora Manual"]
  )

  with pestana_grafico:
    st.markdown(
        "### Comportamiento de la Distribución de Probabilidad"
    )
    st.markdown(
        "El siguiente gráfico muestra la distribución de masa de probabilidad."
        " Modifica los parámetros a la izquierda para analizar cómo cambia la"
        " dispersión y la asimetría."
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
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

  with pestana_validacion:
    st.markdown(
        "### 👨‍💻 Espacio de Validación para Estudiantes"
    )
    st.markdown(
        "Introduce un caso de estudio o valor específico para contrastar tu"
        " procedimiento manual contra el motor de cálculo exacto:"
    )

    col_val1, col_val2 = st.columns([1, 1])

    with col_val1:
      val_ingresado = st.number_input(
          "Seleccione el valor de x a evaluar:",
          min_value=int(min(x)),
          max_value=int(max(x)),
          value=int(media),
          step=1,
      )

      tipo_calculo = st.radio(
          "Tipo de evaluación:",
          [
              "Probabilidad Puntual P(X = x)",
              "Probabilidad Acumulada Menor o Igual P(X ≤ x)",
          ],
      )

    with col_val2:
      st.markdown("#### 📝 Desarrollo y Validación Paso a Paso")
      if distribucion == "Binomial":
        if tipo_calculo == "Probabilidad Puntual P(X = x)":
          comb_val = comb(n, val_ingresado)
          prob_exacta = comb_val * (p**val_ingresado) * (
              (1 - p) ** (n - val_ingresado)
          )
          st.success(
              f"**Evaluación P(X = {val_ingresado}):**\n"
              f"1. Coeficiente Binomial $\\binom{{{n}}}{{{val_ingresado}}} ="
              f" {comb_val}$\n2. Probabilidad de Éxito ($p^{val_ingresado}$) ="
              f" {p**val_ingresado:.6f}\n3. Probabilidad de Fracaso"
              f" ($(1-p)^{n-val_ingresado}$) ="
              f" {((1-p)**(n-val_ingresado)):.6f}\n\n👉 **Resultado Final:**"
              f" **{prob_exacta:.6f}** ({prob_exacta*100:.2f}%)"
          )
        else:
          acumulada = sum(binom_pmf(np.arange(0, val_ingresado + 1), n, p))
          st.success(
              f"**Evaluación Acumulada P(X ≤ {val_ingresado}):**\nSuma de"
              f" probabilidades desde x=0 hasta x={val_ingresado}.\n\n👉"
              f" **Resultado Acumulado:** **{acumulada:.6f}**"
              f" ({acumulada*100:.2f}%)"
          )
      else:  # Poisson
        if tipo_calculo == "Probabilidad Puntual P(X = x)":
          prob_exacta = (np.exp(-mu) * (mu**val_ingresado)) / math.factorial(
              val_ingresado
          )
          st.success(
              f"**Evaluación P(X = {val_ingresado}):**\n"
              f"1. Factor exponencial ($e^{{-{mu}}}$) ="
              f" {np.exp(-mu):.6f}\n2. Término de tasa ($\mu^{val_ingresado}$)"
              f" = {mu**val_ingresado:.6f}\n3. Factorial ($x!$) ="
              f" {math.factorial(val_ingresado)}\n\n👉 **Resultado Final:**"
              f" **{prob_exacta:.6f}** ({prob_exacta*100:.2f}%)"
          )
        else:
          acumulada = sum(poisson_pmf(np.arange(0, val_ingresado + 1), mu))
          st.success(
              f"**Evaluación Acumulada P(X ≤ {val_ingresado}):**\nSuma de"
              f" probabilidades desde x=0 hasta x={val_ingresado}.\n\n👉"
              f" **Resultado Acumulado:** **{acumulada:.6f}**"
              f" ({acumulada*100:.2f}%)"
          )

else:  # Variables Continuas
  distribucion = st.sidebar.selectbox(
      "2. Seleccione el Modelo", ["Normal (Gaussiana)", "Exponencial"]
  )

  if distribucion == "Normal (Gaussiana)":
    st.sidebar.subheader("Parámetros Normales")
    mu = st.sidebar.slider(
        "Media poblacional ($\mu$)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
    )
    sigma = st.sidebar.slider(
        "Desviación Estándar ($\sigma$)",
        min_value=0.1,
        max_value=8.0,
        value=2.0,
        step=0.1,
    )

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    y = norm_pdf(x, mu, sigma)
    media = mu
    varianza = sigma**2
    desv = sigma

    st.subheader(
        f"📌 Modelo Normal