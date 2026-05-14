import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="King County – Análise de Imóveis",
    page_icon="🏠",
    layout="wide",
)

DATA_PATH = r"C:\Users\vish8\OneDrive\Documentos\GitHub\C111-AnaliseDeDados\Trabalho\archive\kc_house_data.csv"

FEATURES = ['sqft_living', 'grade', 'bathrooms', 'bedrooms', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'yr_built', 'lat', 'long']


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.drop(["id", "date", "zipcode"], axis=1, inplace=True)
    # Tratamento
    df.drop_duplicates(inplace=True)
    df = df[(df['bedrooms'] > 0) & (df['bedrooms'] < 20)]
    df = df[df['bathrooms'] > 0]
    q1, q3 = df['price'].quantile(0.25), df['price'].quantile(0.75)
    iqr = q3 - q1
    df = df[(df['price'] >= q1 - 1.5 * iqr) & (df['price'] <= q3 + 1.5 * iqr)]
    df.reset_index(drop=True, inplace=True)
    # Colunas auxiliares
    df["tem_porao"] = (df["sqft_basement"] > 0).astype(int)
    df["renovado"]  = (df["yr_renovated"] > 0).astype(int)
    df["decada"]    = (df["yr_built"] // 10 * 10)
    return df


@st.cache_resource
def train_model(df):
    X = df[FEATURES].copy()
    y = df["price"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return model, scaler, metrics, importances


df = load_data()
model, scaler, metrics, importances = train_model(df)

sns.set_theme(style="whitegrid")

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Navegação")
pagina = st.sidebar.radio(
    "Selecione a seção",
    ["📊 Visão Geral do Dataset",
     "H1 – Waterfront",
     "H2 – Área Habitável",
     "H3 – Grade de Construção",
     "H4 – Renovação",
     "H5 – Banheiros vs Quartos",
     "🤖 Previsão de Preço"],
)

# ════════════════════════════════════════════════════════════════════════════
# VISÃO GERAL
# ════════════════════════════════════════════════════════════════════════════
if pagina == "📊 Visão Geral do Dataset":
    st.title("Mercado Imobiliário de King County, WA")
    st.markdown("Dataset com **21.613 imóveis** vendidos entre 2014 e 2015 — após tratamento: "
                f"**{len(df):,} registros**.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de imóveis", f"{len(df):,}")
    c2.metric("Preço médio", f"${df['price'].mean():,.0f}")
    c3.metric("Preço mínimo", f"${df['price'].min():,.0f}")
    c4.metric("Preço máximo", f"${df['price'].max():,.0f}")

    st.markdown("---")
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Estatísticas descritivas")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribuição das principais variáveis")
    cols_hist = ["price", "sqft_living", "bedrooms", "bathrooms", "grade", "yr_built"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, cols_hist):
        ax.hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Mapa de correlações")
    corr_vars = ["price", "sqft_living", "grade", "bathrooms", "bedrooms",
                 "view", "condition", "floors", "yr_built", "sqft_above"]
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[corr_vars].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax2, annot_kws={"size": 9})
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Distribuição geográfica")
    st.map(df[["lat", "long"]].rename(columns={"lat": "latitude", "long": "longitude"}))

# ════════════════════════════════════════════════════════════════════════════
# H1 – WATERFRONT
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H1 – Waterfront":
    st.title("H1 — Acesso à Orla vs Preço")
    st.markdown("> **Hipótese:** Imóveis com acesso à orla (*waterfront = 1*) têm preço médio "
                "significativamente maior do que os sem orla.")

    media_wf = df.groupby("waterfront")["price"].mean()
    sem, com_ = media_wf[0], media_wf[1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Preço médio SEM orla", f"${sem:,.0f}")
    c2.metric("Preço médio COM orla", f"${com_:,.0f}")
    c3.metric("Diferença", f"+{(com_/sem - 1)*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H1 — Acesso à Orla vs Preço", fontsize=13, fontweight="bold")

    sns.boxplot(data=df, x="waterfront", y="price", palette=["#5b9bd5", "#1f5fa6"], ax=axes[0])
    axes[0].set_xticklabels(["Sem orla", "Com orla"])
    axes[0].set_title("Distribuição de Preços")
    axes[0].set_ylabel("Preço (USD)")
    axes[0].set_xlabel("")

    barras = axes[1].bar(["Sem orla", "Com orla"], media_wf.values,
                          color=["#5b9bd5", "#1f5fa6"], edgecolor="white", width=0.5)
    axes[1].set_title("Preço Médio por Acesso à Orla")
    axes[1].set_ylabel("Preço Médio (USD)")
    for b, v in zip(barras, media_wf.values):
        axes[1].text(b.get_x() + b.get_width()/2, v + 5000,
                     f"${v/1e3:.0f}k", ha="center", fontweight="bold", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H1 CONFIRMADA** — imóveis com orla custam {(com_/sem - 1)*100:.1f}% mais em média.")

# ════════════════════════════════════════════════════════════════════════════
# H2 – ÁREA HABITÁVEL
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H2 – Área Habitável":
    st.title("H2 — Área Habitável vs Preço")
    st.markdown("> **Hipótese:** Quanto maior a área habitável (*sqft_living*), maior o preço.")

    corr = df["sqft_living"].corr(df["price"])
    st.metric("Correlação de Pearson (sqft_living × price)", f"r = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H2 — Área Habitável vs Preço", fontsize=13, fontweight="bold")

    axes[0].scatter(df["sqft_living"], df["price"], alpha=0.2, s=4, color="steelblue")
    coef = np.polyfit(df["sqft_living"], df["price"], 1)
    x_r = np.linspace(df["sqft_living"].min(), df["sqft_living"].max(), 200)
    axes[0].plot(x_r, np.polyval(coef, x_r), color="red", linewidth=2, label="Tendência")
    axes[0].set_title("Dispersão: Área vs Preço")
    axes[0].set_xlabel("Área Habitável (sqft)")
    axes[0].set_ylabel("Preço (USD)")
    axes[0].legend()

    df_tmp = df.copy()
    df_tmp["faixa_area"] = pd.cut(
        df_tmp["sqft_living"],
        bins=[0, 1000, 1500, 2000, 2500, 3000, 4000, 14000],
        labels=["<1k", "1-1.5k", "1.5-2k", "2-2.5k", "2.5-3k", "3-4k", ">4k"],
    )
    media_faixa = df_tmp.groupby("faixa_area", observed=True)["price"].mean()
    cores = sns.color_palette("Blues_r", len(media_faixa))
    barras = axes[1].bar(media_faixa.index, media_faixa.values, color=cores, edgecolor="white")
    axes[1].set_title("Preço Médio por Faixa de Área")
    axes[1].set_xlabel("Faixa de Área (sqft)")
    axes[1].set_ylabel("Preço Médio (USD)")
    axes[1].tick_params(axis="x", rotation=20)
    for b, v in zip(barras, media_faixa.values):
        axes[1].text(b.get_x() + b.get_width()/2, v + 3000,
                     f"${v/1e3:.0f}k", ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H2 CONFIRMADA** — correlação forte: r = {corr:.4f}.")

# ════════════════════════════════════════════════════════════════════════════
# H3 – GRADE
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H3 – Grade de Construção":
    st.title("H3 — Grade de Construção vs Preço")
    st.markdown("> **Hipótese:** O grau de construção (*grade*) é o fator com maior correlação com o preço.")

    corr_vars = ["sqft_living", "grade", "bathrooms", "bedrooms", "view", "condition", "floors", "yr_built"]
    correlacoes = df[corr_vars].corrwith(df["price"]).abs().sort_values(ascending=False)
    pos = list(correlacoes.index).index("grade") + 1

    col1, col2 = st.columns(2)
    col1.metric("Correlação de grade com preço", f"r = {correlacoes['grade']:.4f}")
    col2.metric("Posição no ranking", f"{pos}º lugar")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H3 — Grade de Construção vs Preço", fontsize=13, fontweight="bold")

    cores_rank = ["#d73027" if c == "grade" else "#74add1" for c in correlacoes.index]
    axes[0].barh(correlacoes.index[::-1], correlacoes.values[::-1], color=cores_rank[::-1])
    axes[0].set_title("Correlação das Variáveis com Preço (|r|)")
    axes[0].set_xlabel("|Correlação de Pearson|")
    for i, (var, val) in enumerate(zip(correlacoes.index[::-1], correlacoes.values[::-1])):
        axes[0].text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=9)

    df_grade = df[df["grade"].between(4, 12)]
    sns.boxplot(data=df_grade, x="grade", y="price", palette="YlOrRd", ax=axes[1])
    axes[1].set_title("Distribuição de Preço por Grade (4–12)")
    axes[1].set_xlabel("Grade de Construção")
    axes[1].set_ylabel("Preço (USD)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if pos == 1:
        st.success("✅ **H3 CONFIRMADA** — grade é a variável com maior correlação com o preço.")
    else:
        st.warning(f"⚠️ **H3 PARCIALMENTE CONFIRMADA** — grade ocupa o {pos}º lugar no ranking.")

    st.subheader("Ranking completo")
    st.dataframe(
        correlacoes.reset_index().rename(columns={"index": "Variável", 0: "|r|"})
        .style.format({"|r|": "{:.4f}"}),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════════════════
# H4 – RENOVAÇÃO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H4 – Renovação":
    st.title("H4 — Renovação vs Preço")
    st.markdown("> **Hipótese:** Imóveis que passaram por reforma têm preço médio maior.")

    m_nao = df[df["renovado"] == 0]["price"].mean()
    m_sim = df[df["renovado"] == 1]["price"].mean()
    n_sim = (df["renovado"] == 1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Preço médio NÃO renovado", f"${m_nao:,.0f}")
    c2.metric("Preço médio RENOVADO", f"${m_sim:,.0f}")
    c3.metric("Diferença", f"+{(m_sim/m_nao - 1)*100:.1f}%")
    c4.metric("Imóveis renovados", f"{n_sim:,} ({n_sim/len(df)*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H4 — Renovação vs Preço", fontsize=13, fontweight="bold")

    sns.boxplot(data=df, x="renovado", y="price", palette=["#66c2a5", "#fc8d62"], ax=axes[0])
    axes[0].set_xticklabels(["Não renovado", "Renovado"])
    axes[0].set_title("Distribuição de Preços")
    axes[0].set_ylabel("Preço (USD)")
    axes[0].set_xlabel("")

    for status, label, cor in [(0, "Não renovado", "#66c2a5"), (1, "Renovado", "#fc8d62")]:
        dados = df[df["renovado"] == status]["price"]
        axes[1].hist(dados, bins=60, alpha=0.55,
                     label=f"{label} (n={len(dados):,})", color=cor, density=True)
    axes[1].axvline(m_nao, color="#3aa882", linestyle="--", linewidth=2,
                    label=f"Média não renov.: ${m_nao/1e3:.0f}k")
    axes[1].axvline(m_sim, color="#e05a20", linestyle="--", linewidth=2,
                    label=f"Média renovado: ${m_sim/1e3:.0f}k")
    axes[1].set_title("Distribuição de Preços (densidade)")
    axes[1].set_xlabel("Preço (USD)")
    axes[1].set_ylabel("Densidade")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H4 CONFIRMADA** — renovados custam {(m_sim/m_nao - 1)*100:.1f}% mais em média.")

# ════════════════════════════════════════════════════════════════════════════
# H5 – BANHEIROS VS QUARTOS
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H5 – Banheiros vs Quartos":
    st.title("H5 — Banheiros vs Quartos: correlação com o preço")
    st.markdown("> **Hipótese:** O número de banheiros tem correlação mais forte com o preço do que quartos.")

    r_bath = df["bathrooms"].corr(df["price"])
    r_bed  = df["bedrooms"].corr(df["price"])

    c1, c2 = st.columns(2)
    c1.metric("Correlação – Banheiros", f"r = {r_bath:.4f}")
    c2.metric("Correlação – Quartos",   f"r = {r_bed:.4f}",
              delta=f"{r_bed - r_bath:+.4f} vs banheiros")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H5 — Banheiros vs Quartos", fontsize=13, fontweight="bold")

    df_bath = df[df["bathrooms"] <= 6]
    media_bath = df_bath.groupby("bathrooms")["price"].mean()
    axes[0].scatter(media_bath.index, media_bath.values, s=100, color="coral", zorder=5)
    coef_b = np.polyfit(df_bath["bathrooms"], df_bath["price"], 1)
    x_b = np.linspace(df_bath["bathrooms"].min(), df_bath["bathrooms"].max(), 100)
    axes[0].plot(x_b, np.polyval(coef_b, x_b), "r--", linewidth=2)
    axes[0].set_title(f"Banheiros vs Preço Médio  (r = {r_bath:.3f})")
    axes[0].set_xlabel("Número de Banheiros")
    axes[0].set_ylabel("Preço Médio (USD)")

    df_bed = df[df["bedrooms"].between(1, 8)]
    media_bed = df_bed.groupby("bedrooms")["price"].mean()
    axes[1].scatter(media_bed.index, media_bed.values, s=100, color="steelblue", zorder=5)
    coef_r = np.polyfit(df_bed["bedrooms"], df_bed["price"], 1)
    x_r = np.linspace(1, 8, 100)
    axes[1].plot(x_r, np.polyval(coef_r, x_r), "b--", linewidth=2)
    axes[1].set_title(f"Quartos vs Preço Médio  (r = {r_bed:.3f})")
    axes[1].set_xlabel("Número de Quartos")
    axes[1].set_ylabel("Preço Médio (USD)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if r_bath > r_bed:
        st.success(f"✅ **H5 CONFIRMADA** — banheiros têm correlação mais forte (r = {r_bath:.4f} > {r_bed:.4f}).")
    else:
        st.error(f"❌ **H5 REJEITADA** — quartos têm correlação mais forte (r = {r_bed:.4f} > {r_bath:.4f}).")

# ════════════════════════════════════════════════════════════════════════════
# PREVISÃO DE PREÇO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Previsão de Preço":
    st.title("🤖 Previsão de Preço — Random Forest")
    st.markdown("Modelo treinado com os dados tratados de King County. "
                "Ajuste as características do imóvel e veja o preço estimado.")

    # Métricas do modelo
    st.subheader("Desempenho do modelo (conjunto de teste — 20%)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{metrics['r2']:.4f}", help="Quanto maior, melhor (máx = 1)")
    c2.metric("MAE", f"${metrics['mae']:,.0f}", help="Erro médio absoluto")
    c3.metric("RMSE", f"${metrics['rmse']:,.0f}", help="Raiz do erro quadrático médio")

    # Importância das features
    with st.expander("📊 Ver importância das variáveis"):
        fig_imp, ax_imp = plt.subplots(figsize=(9, 4))
        importances.plot(kind="bar", ax=ax_imp, color="steelblue", edgecolor="white")
        ax_imp.set_title("Importância das Features")
        ax_imp.set_ylabel("Importância")
        ax_imp.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

    st.markdown("---")
    st.subheader("Configurar o imóvel")

    col1, col2, col3 = st.columns(3)

    with col1:
        sqft_living = st.slider("Área habitável (sqft)", 300, 8000, 1800, step=50)
        sqft_above  = st.slider("Área acima do solo (sqft)", 300, 8000, 1500, step=50)
        grade       = st.slider("Grade de construção (1–13)", 1, 13, 7)
        condition   = st.slider("Condição (1–5)", 1, 5, 3)

    with col2:
        bedrooms    = st.slider("Quartos", 1, 10, 3)
        bathrooms   = st.slider("Banheiros", 1, 8, 2)
        floors      = st.selectbox("Andares", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=0)
        yr_built    = st.slider("Ano de construção", 1900, 2015, 1975)

    with col3:
        waterfront  = st.selectbox("Acesso à orla", ["Não", "Sim"])
        view        = st.slider("Índice de vista (0–4)", 0, 4, 0)
        lat         = st.number_input("Latitude", value=47.56, min_value=47.15, max_value=47.78, step=0.01)
        long        = st.number_input("Longitude", value=-122.21, min_value=-122.52, max_value=-121.31, step=0.01)

    waterfront_val = 1 if waterfront == "Sim" else 0

    entrada = np.array([[sqft_living, grade, bathrooms, bedrooms, floors,
                         waterfront_val, view, condition, sqft_above, yr_built, lat, long]])
    entrada_scaled = scaler.transform(entrada)
    preco_previsto = model.predict(entrada_scaled)[0]

    st.markdown("---")
    st.markdown(
        f"<h2 style='text-align:center; color:#1f5fa6;'>💰 Preço estimado: "
        f"<strong>${preco_previsto:,.0f}</strong></h2>",
        unsafe_allow_html=True,
    )

    # Comparação com a média do dataset
    media_geral = df["price"].mean()
    diff_pct = (preco_previsto / media_geral - 1) * 100
    simbolo = "acima" if diff_pct >= 0 else "abaixo"
    st.caption(f"Preço médio do dataset: ${media_geral:,.0f} — "
               f"estimativa {abs(diff_pct):.1f}% {simbolo} da média.")
