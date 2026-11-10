"""

Simulador y Laboratorio Probabilístico — UNAD
 
Ejecutar:  streamlit run app_unad.py
 
Opcional (recomendado): crear .streamlit/config.toml para que radios,

sliders y enlaces de Streamlit también usen la paleta institucional:
 
    [theme]

    primaryColor = "#005A9C"

    backgroundColor = "#FFFFFF"

    secondaryBackgroundColor = "#EAF7FB"

    textColor = "#1B365D"

"""
 
import math
 
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import streamlit as st

from matplotlib.patches import Patch

from matplotlib.ticker import MaxNLocator
 
# ==========================================

# PALETA INSTITUCIONAL

# ==========================================

AZUL_OSCURO = "#1B365D"

AZUL = "#005A9C"

CELESTE = "#60C1E5"

CELESTE_CLARO = "#A0E0EF"
 
# ==========================================

# CONFIGURACIÓN DE PÁGINA Y ESTILOS

# ==========================================

st.set_page_config(

    page_title="Laboratorio Probabilístico - UNAD",

    layout="wide",

    initial_sidebar_state="expanded",

    page_icon="🎓",

)
 
st.markdown(

    """
<style>

    .block-container { padding-top: 1.8rem; max-width: 1200px; }

    #MainMenu, footer { visibility: hidden; }
 
    h1, h2, h3, h4 { color: #1B365D !important; font-weight: 700; }
 
    /* ---------- Barra lateral ---------- */

    [data-testid="stSidebar"] {

        background: linear-gradient(180deg, #FFFFFF 0%, rgba(160,224,239,0.32) 100%);

        border-right: 3px solid #005A9C;

    }

    .brand {

        background: #1B365D;

        border-left: 6px solid #60C1E5;

        border-radius: 8px;

        padding: 0.8rem 1rem;

        margin-bottom: 1rem;

    }

    .brand-title { color: #FFFFFF; font-size: 1.35rem; font-weight: 800; line-height: 1.1; }

    .brand-sub { color: #A0E0EF; font-size: 0.82rem; margin-top: 0.2rem; }
 
    /* ---------- Encabezado principal ---------- */

    .hero {

        background: linear-gradient(120deg, #1B365D 0%, #005A9C 100%);

        border-bottom: 5px solid #60C1E5;

        border-radius: 14px;

        padding: 1.7rem 2rem;

        margin-bottom: 1.1rem;

    }

    .hero-title { color: #FFFFFF; font-size: 2rem; font-weight: 800; line-height: 1.15; }

    .hero-sub { color: #A0E0EF; font-size: 0.97rem; margin-top: 0.5rem; max-width: 70ch; }
 
    /* ---------- Etiquetas de contexto ---------- */

    .badge {

        display: inline-block;

        background: #A0E0EF;

        color: #1B365D;

        font-weight: 700;

        font-size: 0.85rem;

        padding: 0.25rem 0.85rem;

        border-radius: 999px;

        margin: 0 0.4rem 0.9rem 0;

    }

    .badge.oscuro { background: #1B365D; color: #FFFFFF; }

    .badge.medio { background: #005A9C; color: #FFFFFF; }
 
    /* ---------- Métricas ---------- */

    [data-testid="stMetric"] {

        background: #FFFFFF;

        border: 1px solid #A0E0EF;

        border-left: 6px solid #005A9C;

        border-radius: 8px;

        padding: 0.7rem 1rem;

    }

    [data-testid="stMetricLabel"] p { color: #005A9C; font-weight: 600; }

    [data-testid="stMetricValue"] { color: #1B365D; font-weight: 800; }
 
    /* ---------- Tarjeta de resultado (elemento protagonista) ---------- */

    .result-card {

        background: linear-gradient(135deg, #005A9C 0%, #1B365D 100%);

        border-radius: 12px;

        padding: 1.2rem 1.3rem;

        margin-top: 0.6rem;

        border-bottom: 4px solid #60C1E5;

    }

    .result-label { color: #A0E0EF; font-size: 0.95rem; font-weight: 600; }

    .result-value { color: #FFFFFF; font-size: 2.3rem; font-weight: 800; line-height: 1.15; }

    .result-sub { color: #60C1E5; font-size: 0.88rem; margin-top: 0.2rem; }
 
    /* ---------- Pestañas ---------- */

    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #A0E0EF; }

    .stTabs [data-baseweb="tab"] { font-weight: 600; color: #1B365D; padding: 0.6rem 1rem; }

    .stTabs [aria-selected="true"] { color: #005A9C; }

    .stTabs [data-baseweb="tab-highlight"] { background-color: #005A9C; }
 
    /* ---------- Botones ---------- */

    .stButton > button {

        background-color: #005A9C;

        color: white;

        border-radius: 6px;

        border: none;

        font-weight: bold;

        width: 100%;

    }

    .stButton > button:hover { background-color: #1B365D; color: white; }
</style>

    """,

    unsafe_allow_html=True,

)
 
# ==========================================

# FUNCIONES MATEMÁTICAS

# ==========================================

DISC_MODOS = ["P(X = x)", "P(X ≤ x)", "P(X ≥ x)"]

CONT_MODOS = ["f(x)  ·  densidad", "P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)"]
 
 
def pmf_binomial(k: np.ndarray, n: int, p: float) -> np.ndarray:

    return np.array(

        [math.comb(n, int(i)) * (p ** int(i)) * ((1 - p) ** (n - int(i))) for i in k]

    )
 
 
def pmf_poisson(k: np.ndarray, mu: float) -> np.ndarray:

    # Se calcula en escala logarítmica para evitar desbordes con k grande

    return np.array(

        [math.exp(-mu + int(i) * math.log(mu) - math.lgamma(int(i) + 1)) for i in k]

    )
 
 
def pdf_normal(x, mu: float, sigma: float):

    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
 
 
def cdf_normal(x: float, mu: float, sigma: float) -> float:

    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))
 
 
def pdf_exponencial(x, lam: float):

    return lam * np.exp(-lam * x)
 
 
def cdf_exponencial(x: float, lam: float) -> float:

    return 1 - math.exp(-lam * x) if x >= 0 else 0.0
 
 
def fmt(v, dec: int = 6) -> str:

    """Formato numérico legible (miles, decimales o notación científica)."""

    if isinstance(v, int):

        return f"{v:,}" if abs(v) < 10**12 else f"{float(v):.4e}"

    v = float(v)

    if v == 0:

        return "0"

    if abs(v) >= 1e6 or abs(v) < 1e-4:

        return f"{v:.4e}"

    return f"{v:.{dec}f}"
 
 
# ==========================================

# COMPONENTES VISUALES

# ==========================================

def tarjeta_resultado(etiqueta: str, valor: str, detalle: str = ""):

    st.markdown(

        f"""
<div class="result-card">
<div class="result-label">{etiqueta}</div>
<div class="result-value">{valor}</div>
<div class="result-sub">{detalle}</div>
</div>

        """,

        unsafe_allow_html=True,

    )
 
 
def fila_metricas(media: float, varianza: float, desv: float):

    c1, c2, c3 = st.columns(3)

    c1.metric("Esperanza E[X]", f"{media:.4f}")

    c2.metric("Varianza V[X]", f"{varianza:.4f}")

    c3.metric("Desviación estándar σ", f"{desv:.4f}")
 
 
def mostrar_tabla(filas, columnas):

    st.dataframe(pd.DataFrame(filas, columns=columnas), hide_index=True)
 
 
def _estilo_ejes(ax, xlabel: str, ylabel: str, titulo: str):

    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", color=AZUL_OSCURO)

    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color=AZUL_OSCURO)

    ax.set_title(titulo, fontsize=12, fontweight="bold", color=AZUL_OSCURO, loc="left", pad=12)

    ax.grid(axis="y", linestyle="--", alpha=0.45, color=CELESTE_CLARO)

    ax.set_axisbelow(True)

    for lado in ("top", "right"):

        ax.spines[lado].set_visible(False)

    for lado in ("left", "bottom"):

        ax.spines[lado].set_color(CELESTE)

    ax.tick_params(colors=AZUL_OSCURO, labelsize=9)
 
 
def graficar_discreta(x, y, titulo: str, mascara):

    fig, ax = plt.subplots(figsize=(9, 4.4))

    colores = [AZUL if m else CELESTE_CLARO for m in mascara]

    bordes = [AZUL_OSCURO if m else CELESTE for m in mascara]

    ax.bar(x, y, color=colores, edgecolor=bordes, linewidth=0.9, width=0.8)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    _estilo_ejes(ax, "Valores de la variable (x)", "Probabilidad P(X = x)", titulo)

    ax.legend(

        handles=[

            Patch(facecolor=AZUL, edgecolor=AZUL_OSCURO, label="Región seleccionada"),

            Patch(facecolor=CELESTE_CLARO, edgecolor=CELESTE, label="Resto de la distribución"),

        ],

        frameon=False,

        fontsize=9,

        labelcolor=AZUL_OSCURO,

    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
 
 
def graficar_continua(x, y, titulo: str, mascara=None, x_marca=None, y_marca=None):

    fig, ax = plt.subplots(figsize=(9, 4.4))

    ax.fill_between(x, y, color=CELESTE_CLARO, alpha=0.55, label="Distribución")

    if mascara is not None and mascara.any():

        ax.fill_between(

            x, y, where=mascara, color=AZUL, alpha=0.7, interpolate=True,

            label="Región seleccionada",

        )

    ax.plot(x, y, color=AZUL_OSCURO, linewidth=2.5, label="f(x)")

    if x_marca is not None and x[0] <= x_marca <= x[-1]:

        ax.axvline(x_marca, color=AZUL, linestyle="--", linewidth=1.3)

        ax.scatter([x_marca], [y_marca], color=AZUL_OSCURO, s=55, zorder=5, label="Punto evaluado")

    ax.set_ylim(bottom=0)

    _estilo_ejes(ax, "Valores (x)", "Densidad f(x)", titulo)

    ax.grid(axis="x", linestyle="--", alpha=0.25, color=CELESTE_CLARO)

    ax.legend(frameon=False, fontsize=9, labelcolor=AZUL_OSCURO)

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)
 
 
# ==========================================

# CÁLCULOS PASO A PASO (filas de la tabla de validación)

# ==========================================

def filas_binomial(n: int, p: float, k: int):

    c = math.comb(n, k)

    a = p**k

    b = (1 - p) ** (n - k)

    filas = [

        ("Coeficiente combinatorio", f"C({n}, {k})", fmt(c)),

        ("Probabilidad de los éxitos", f"{p:g}^{k}", fmt(a)),

        ("Probabilidad de los fracasos", f"(1 − {p:g})^{n - k}", fmt(b)),

        (f"Resultado P(X = {k})", "C · p^k · (1 − p)^(n − k)", fmt(c * a * b)),

    ]

    return filas
 
 
def filas_poisson(mu: float, k: int):

    e = math.exp(-mu)

    t = mu**k

    f = math.factorial(k)

    pk = math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))

    return [

        ("Constante exponencial", f"e^(−{mu:g})", fmt(e)),

        ("Término de la tasa", f"{mu:g}^{k}", fmt(t)),

        ("Factorial", f"{k}!", fmt(f)),

        (f"Resultado P(X = {k})", "e^(−μ) · μ^k / k!", fmt(pk)),

    ]
 
 
def filas_pdf_normal(mu: float, sigma: float, xv: float):

    z = (xv - mu) / sigma

    coef = 1 / (sigma * math.sqrt(2 * math.pi))

    expo = math.exp(-0.5 * z * z)

    return [

        ("Valor estandarizado", "z = (x − μ) / σ", fmt(z)),

        ("Constante de normalización", "1 / (σ · √(2π))", fmt(coef)),

        ("Exponencial de la campana", "e^(−z² / 2)", fmt(expo)),

        ("Densidad f(x)", "constante · exponencial", fmt(coef * expo)),

    ]
 
 
def filas_cdf_normal(mu: float, sigma: float, xv: float):

    z = (xv - mu) / sigma

    return [

        ("Valor estandarizado", "z = (x − μ) / σ", fmt(z)),

        ("Probabilidad acumulada", "F(x) = Φ(z)", fmt(cdf_normal(xv, mu, sigma))),

    ]
 
 
def filas_pdf_exp(lam: float, xv: float):

    return [

        ("Argumento exponencial", "−λ · x", fmt(-lam * xv)),

        ("Exponencial", "e^(−λ · x)", fmt(math.exp(-lam * xv))),

        ("Densidad f(x)", "λ · e^(−λ · x)", fmt(lam * math.exp(-lam * xv))),

    ]
 
 
def filas_cdf_exp(lam: float, xv: float):

    return [

        ("Argumento exponencial", "−λ · x", fmt(-lam * xv)),

        ("Cola derecha", "e^(−λ · x) = P(X ≥ x)", fmt(math.exp(-lam * xv))),

        ("Probabilidad acumulada", "F(x) = 1 − e^(−λ · x)", fmt(cdf_exponencial(xv, lam))),

    ]
 
 
# ==========================================

# BLOQUES DE INTERFAZ COMPARTIDOS

# ==========================================

def render_teoria(cfg: dict):

    st.markdown(f"#### 📚 {cfg['nombre']}")

    for expresion in cfg["latex"]:

        st.latex(expresion)

    st.markdown(cfg["teoria"])
 
 
def render_discreta(cfg: dict):

    x, y = cfg["x"], cfg["y"]

    tab_exp, tab_pasos, tab_teoria = st.tabs(

        ["📈 Explorador", "🧮 Validación paso a paso", "📚 Teoría"]

    )
 
    with tab_exp:

        col_ctrl, col_graf = st.columns([1, 2], gap="large")

        with col_ctrl:

            st.markdown("#### 🎯 Calculadora")

            modo = st.radio("Tipo de probabilidad", DISC_MODOS)

            x_sel = int(

                st.number_input(

                    "Valor de x",

                    min_value=0,

                    max_value=int(cfg["x_max"]),

                    value=int(cfg["x_def"]),

                    step=1,

                )

            )

            if modo == DISC_MODOS[0]:

                mascara = x == x_sel

                prob = float(y[mascara].sum())

            elif modo == DISC_MODOS[1]:

                mascara = x <= x_sel

                prob = float(y[mascara].sum())

            else:

                mascara = x >= x_sel

                prob = float(1 - y[x < x_sel].sum())

            prob = min(max(prob, 0.0), 1.0)

            etiqueta = modo.replace("x", str(x_sel))

            tarjeta_resultado(etiqueta, f"{prob:.6f}", f"{prob * 100:.2f} %")

        with col_graf:

            graficar_discreta(x, y, cfg["titulo_grafica"], mascara)
 
    with tab_pasos:

        st.markdown("#### 🔍 Validación manual para estudiantes")

        st.latex(cfg["latex"][0])

        st.markdown(f"**Cálculo de P(X = {x_sel})**")

        mostrar_tabla(cfg["filas"](x_sel), ["Componente", "Expresión", "Valor"])
 
        if modo != DISC_MODOS[0]:

            if modo == DISC_MODOS[1]:

                st.latex(r"P(X \le x) = \sum_{k=0}^{x} P(X = k)")

                ks = x[x <= x_sel]

            else:

                st.latex(r"P(X \ge x) = 1 - \sum_{k=0}^{x-1} P(X = k)")

                ks = x[x < x_sel]

            if len(ks) == 0:

                st.info(f"No hay términos que restar: {etiqueta} = 1.")

            else:

                probs = y[: len(ks)]

                acum = np.cumsum(probs)

                filas = [(int(k), fmt(pk), fmt(a)) for k, pk, a in zip(ks, probs, acum)]

                st.markdown("**Términos de la suma**")

                mostrar_tabla(filas, ["k", "P(X = k)", "Acumulado"])

        st.success(f"Resultado: {etiqueta} = {prob:.6f}  ({prob * 100:.2f} %)")
 
    with tab_teoria:

        render_teoria(cfg)
 
 
def render_continua(cfg: dict):

    x, y = cfg["x"], cfg["y"]

    pdf, cdf = cfg["pdf"], cfg["cdf"]

    tab_exp, tab_pasos, tab_teoria = st.tabs(

        ["📈 Explorador", "🧮 Validación paso a paso", "📚 Teoría"]

    )
 
    with tab_exp:

        col_ctrl, col_graf = st.columns([1, 2], gap="large")

        with col_ctrl:

            st.markdown("#### 🎯 Calculadora")

            modo = st.radio("Qué desea calcular", CONT_MODOS)

            minimo = cfg["x_min"]

            x_sel = a = b = None
 
            if modo == CONT_MODOS[3]:

                a = st.number_input(

                    "Límite inferior (a)", min_value=minimo,

                    value=float(cfg["a_def"]), step=0.1, format="%.3f",

                )

                b = st.number_input(

                    "Límite superior (b)", min_value=minimo,

                    value=float(cfg["b_def"]), step=0.1, format="%.3f",

                )

                if a > b:

                    st.warning("a era mayor que b: se intercambiaron los límites.")

                    a, b = b, a

            else:

                x_sel = st.number_input(

                    "Valor de x", min_value=minimo,

                    value=float(cfg["x_def"]), step=0.1, format="%.3f",

                )
 
            mascara = None

            x_marca = y_marca = None

            if modo == CONT_MODOS[0]:

                valor = float(pdf(x_sel))

                etiqueta = f"f({x_sel:.3f})"

                detalle = "Densidad en el punto (no es una probabilidad)"

                x_marca, y_marca = x_sel, valor

            elif modo == CONT_MODOS[1]:

                valor = cdf(x_sel)

                etiqueta = f"P(X ≤ {x_sel:.3f})"

                detalle = f"{valor * 100:.2f} %"

                mascara = x <= x_sel

            elif modo == CONT_MODOS[2]:

                valor = 1 - cdf(x_sel)

                etiqueta = f"P(X ≥ {x_sel:.3f})"

                detalle = f"{valor * 100:.2f} %"

                mascara = x >= x_sel

            else:

                valor = cdf(b) - cdf(a)

                etiqueta = f"P({a:.3f} ≤ X ≤ {b:.3f})"

                detalle = f"{valor * 100:.2f} %"

                mascara = (x >= a) & (x <= b)

            tarjeta_resultado(etiqueta, f"{valor:.6f}", detalle)

        with col_graf:

            graficar_continua(x, y, cfg["titulo_grafica"], mascara, x_marca, y_marca)
 
    with tab_pasos:

        st.markdown("#### 🔍 Validación manual para estudiantes")

        st.latex(cfg["latex"][0])

        cols = ["Componente", "Expresión", "Valor"]
 
        def con_punto(filas, nombre):

            return [(f"{c} ({nombre})", e, v) for c, e, v in filas]
 
        if modo == CONT_MODOS[0]:

            mostrar_tabla(cfg["filas_pdf"](x_sel), cols)

        elif modo == CONT_MODOS[1]:

            mostrar_tabla(cfg["filas_cdf"](x_sel), cols)

        elif modo == CONT_MODOS[2]:

            filas = cfg["filas_cdf"](x_sel) + [

                ("Cola derecha", "P(X ≥ x) = 1 − F(x)", fmt(valor))

            ]

            mostrar_tabla(filas, cols)

        else:

            filas = (

                con_punto(cfg["filas_cdf"](a), "a")

                + con_punto(cfg["filas_cdf"](b), "b")

                + [("Intervalo", "P = F(b) − F(a)", fmt(valor))]

            )

            mostrar_tabla(filas, cols)
 
        if modo == CONT_MODOS[0]:

            st.success(f"Resultado: {etiqueta} = {valor:.6f}")

        else:

            st.success(f"Resultado: {etiqueta} = {valor:.6f}  ({valor * 100:.2f} %)")
 
    with tab_teoria:

        render_teoria(cfg)
 
 
# ==========================================

# ENCABEZADO Y BARRA LATERAL

# ==========================================

st.markdown(

    """
<div class="hero">
<div class="hero-title">Simulador y Validador de Variables Aleatorias</div>
<div class="hero-sub">

            Universidad Nacional Abierta y a Distancia (UNAD). Explore distribuciones,

            calcule probabilidades y compruebe a mano cada resultado.
</div>
</div>

    """,

    unsafe_allow_html=True,

)
 
st.sidebar.markdown(

    """
<div class="brand">
<div class="brand-title">🎓 UNAD</div>
<div class="brand-sub">Laboratorio probabilístico</div>
</div>

    """,

    unsafe_allow_html=True,

)
 
tipo_variable = st.sidebar.selectbox(

    "Tipo de variable aleatoria",

    ["Discreta", "Continua"],

    help="Seleccione el tipo de variable para configurar el simulador.",

)

st.sidebar.markdown("### ⚙️ Configuración")
 
# ==========================================

# VARIABLES DISCRETAS

# ==========================================

if tipo_variable == "Discreta":

    distribucion = st.sidebar.radio(

        "Modelo de distribución",

        ["Binomial", "Poisson"],

        help="Seleccione el modelo de distribución discreta.",

    )

    st.sidebar.markdown("### 🔢 Parámetros")
 
    if distribucion == "Binomial":

        n = int(st.sidebar.number_input("Número de ensayos (n)", min_value=1, max_value=500, value=10, step=1))

        p = st.sidebar.number_input("Probabilidad de éxito (p)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
 
        x = np.arange(0, n + 1)

        y = pmf_binomial(x, n, p)

        notacion = f"X ~ Bin(n={n}, p={p:g})"

        st.markdown(

            f'<span class="badge oscuro">Discreta</span>'

            f'<span class="badge medio">Binomial</span>'

            f'<span class="badge">{notacion}</span>',

            unsafe_allow_html=True,

        )

        fila_metricas(n * p, n * p * (1 - p), math.sqrt(n * p * (1 - p)))

        st.write("")

        render_discreta(

            {

                "nombre": "Distribución Binomial",

                "x": x,

                "y": y,

                "x_max": n,

                "x_def": min(n, round(n * p)),

                "titulo_grafica": f"Distribución Binomial  n={n}, p={p:g}",

                "filas": lambda k: filas_binomial(n, p, k),

                "latex": [

                    r"P(X = k) = \binom{n}{k}\, p^{k}\,(1 - p)^{n - k}",

                    r"E[X] = np \qquad V[X] = np(1 - p)",

                ],

                "teoria": (

                    "Modela el **número de éxitos** en *n* ensayos independientes, "

                    "cada uno con la misma probabilidad de éxito *p*."

                ),

            }

        )

    else:

        mu = st.sidebar.number_input("Tasa media (μ)", min_value=0.01, max_value=50.0, value=5.0, step=0.1)
 
        x_max = int(mu * 3) + 15

        x = np.arange(0, x_max)

        y = pmf_poisson(x, mu)

        notacion = f"X ~ Pois(μ={mu:g})"

        st.markdown(

            f'<span class="badge oscuro">Discreta</span>'

            f'<span class="badge medio">Poisson</span>'

            f'<span class="badge">{notacion}</span>',

            unsafe_allow_html=True,

        )

        fila_metricas(mu, mu, math.sqrt(mu))

        st.write("")

        render_discreta(

            {

                "nombre": "Distribución de Poisson",

                "x": x,

                "y": y,

                "x_max": x_max - 1,

                "x_def": int(mu),

                "titulo_grafica": f"Distribución de Poisson  μ={mu:g}",

                "filas": lambda k: filas_poisson(mu, k),

                "latex": [

                    r"P(X = k) = \dfrac{e^{-\mu}\,\mu^{k}}{k!}",

                    r"E[X] = V[X] = \mu",

                ],

                "teoria": (

                    "Modela el **número de eventos** que ocurren en un intervalo fijo "

                    "de tiempo o espacio, cuando ocurren de forma independiente y con "

                    "una tasa media constante *μ*."

                ),

            }

        )
 
# ==========================================

# VARIABLES CONTINUAS

# ==========================================

else:

    distribucion = st.sidebar.radio(

        "Modelo de distribución",

        ["Normal (Gaussiana)", "Exponencial"],

        help="Seleccione el modelo de distribución continua.",

    )

    st.sidebar.markdown("### 📈 Parámetros")
 
    if distribucion == "Normal (Gaussiana)":

        mu = st.sidebar.number_input("Media (μ)", value=0.0, step=0.1)

        sigma = st.sidebar.number_input("Desviación estándar (σ)", min_value=0.1, value=1.5, step=0.1)
 
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)

        y = pdf_normal(x, mu, sigma)

        notacion = f"X ~ N(μ={mu:g}, σ²={sigma**2:g})"

        st.markdown(

            f'<span class="badge oscuro">Continua</span>'

            f'<span class="badge medio">Normal</span>'

            f'<span class="badge">{notacion}</span>',

            unsafe_allow_html=True,

        )

        fila_metricas(mu, sigma**2, sigma)

        st.write("")

        render_continua(

            {

                "nombre": "Distribución Normal",

                "x": x,

                "y": y,

                "x_min": None,

                "x_def": mu,

                "a_def": mu - sigma,

                "b_def": mu + sigma,

                "pdf": lambda v: pdf_normal(v, mu, sigma),

                "cdf": lambda v: cdf_normal(v, mu, sigma),

                "titulo_grafica": f"Distribución Normal  μ={mu:g}, σ={sigma:g}",

                "filas_pdf": lambda v: filas_pdf_normal(mu, sigma, v),

                "filas_cdf": lambda v: filas_cdf_normal(mu, sigma, v),

                "latex": [

                    r"f(x) = \dfrac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^{2}}",

                    r"z = \dfrac{x - \mu}{\sigma} \qquad F(x) = \Phi(z)",

                ],

                "teoria": (

                    "Distribución simétrica en forma de campana. Aproximadamente "

                    "**68 %** de los valores cae a ±1σ de la media, **95 %** a ±2σ "

                    "y **99.7 %** a ±3σ."

                ),

            }

        )

    else:

        lam = st.sidebar.number_input("Parámetro de tasa (λ)", min_value=0.1, value=1.0, step=0.1)
 
        x = np.linspace(0, 5 / lam, 400)

        y = pdf_exponencial(x, lam)

        notacion = f"X ~ Exp(λ={lam:g})"

        st.markdown(

            f'<span class="badge oscuro">Continua</span>'

            f'<span class="badge medio">Exponencial</span>'

            f'<span class="badge">{notacion}</span>',

            unsafe_allow_html=True,

        )

        fila_metricas(1 / lam, 1 / lam**2, 1 / lam)

        st.write("")

        render_continua(

            {

                "nombre": "Distribución Exponencial",

                "x": x,

                "y": y,

                "x_min": 0.0,

                "x_def": 1 / lam,

                "a_def": 0.5 / lam,

                "b_def": 2 / lam,

                "pdf": lambda v: lam * math.exp(-lam * v),

                "cdf": lambda v: cdf_exponencial(v, lam),

                "titulo_grafica": f"Distribución Exponencial  λ={lam:g}",

                "filas_pdf": lambda v: filas_pdf_exp(lam, v),

                "filas_cdf": lambda v: filas_cdf_exp(lam, v),

                "latex": [

                    r"f(x) = \lambda\, e^{-\lambda x}, \quad x \ge 0",

                    r"F(x) = 1 - e^{-\lambda x} \qquad E[X] = \dfrac{1}{\lambda}",

                ],

                "teoria": (

                    "Modela el **tiempo de espera** hasta el próximo evento en un "

                    "proceso de Poisson. Tiene la propiedad de *falta de memoria*: "

                    "lo que ya esperó no cambia lo que falta."

                ),

            }

        )
 
st.markdown("---")

st.caption("Universidad Nacional Abierta y a Distancia · Laboratorio de probabilidad y estadística")
 
