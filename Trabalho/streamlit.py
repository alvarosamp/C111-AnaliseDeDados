import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from modelo_ml import load_data, train_model, FEATURES

st.set_page_config(
    page_title="King County – Análise de Imóveis",
    page_icon="🏠",
    layout="wide",
)

sns.set_theme(style="whitegrid")


@st.cache_data
def get_data():
    return load_data()


@st.cache_resource
def get_model(df):
    return train_model(df)


df = get_data()
model, scaler, metrics, importances = get_model(df)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 Navegação")
pagina = st.sidebar.radio(
    "Selecione a seção",
    [
        "📊 Introdução",
        "H1 – Área vs Preço",
        "H2 – Grade vs Preço",
        "H3 – Waterfront vs Preço",
        "H4 – Banheiros vs Quartos",
        "H5 – Reformados são mais antigos?",
        "H6 – Waterfront e Vista",
        "H7 – Localização e Padrão",
        "H8 – Condição vs Grade",
        "H9 – Andares vs Área",
        "H10 – Lote vs Área Construída",
        "🤖 Previsão de Preço",
    ],
)

# ════════════════════════════════════════════════════════════════════════════
# INTRODUÇÃO
# ════════════════════════════════════════════════════════════════════════════
if pagina == "📊 Introdução":
    st.title("Mercado Imobiliário de King County, WA")
    st.markdown(
        """
        ### Sobre o dataset
        O **KC House Data** reúne **21.613 transações** de imóveis residenciais
        ocorridas no condado de King (Seattle, EUA) entre **maio de 2014 e maio de 2015**.
        É amplamente usado em estudos de regressão e análise de preços imobiliários.

        **Após tratamento** (remoção de duplicatas, outliers de preço e registros inválidos):
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de imóveis", f"{len(df):,}")
    c2.metric("Preço médio", f"${df['price'].mean():,.0f}")
    c3.metric("Preço mínimo", f"${df['price'].min():,.0f}")
    c4.metric("Preço máximo", f"${df['price'].max():,.0f}")

    st.markdown("---")
    st.subheader("Variáveis do dataset")
    variaveis = {
        "price": "Preço de venda (USD)",
        "bedrooms": "Número de quartos",
        "bathrooms": "Número de banheiros",
        "sqft_living": "Área habitável (sqft)",
        "sqft_lot": "Área do terreno (sqft)",
        "floors": "Número de andares",
        "waterfront": "Acesso à orla (0/1)",
        "view": "Vista (0–4)",
        "condition": "Condição do imóvel (1–5)",
        "grade": "Padrão de construção (1–13)",
        "sqft_above": "Área acima do solo (sqft)",
        "sqft_basement": "Área do porão (sqft)",
        "lat / long": "Coordenadas geográficas",
        "Years": "Idade do imóvel (anos)",
        "yrs_renovated": "Anos desde a última reforma (0 = nunca reformado)",
    }
    st.table(pd.DataFrame(variaveis.items(), columns=["Coluna", "Descrição"]))

    st.markdown("---")
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Estatísticas descritivas")
    cols_desc = ["price", "sqft_living", "bedrooms", "bathrooms", "grade",
                 "condition", "floors", "Years", "yrs_renovated"]
    st.dataframe(df[cols_desc].describe().T.style.format("{:.2f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribuição das principais variáveis")
    cols_hist = ["price", "sqft_living", "bedrooms", "bathrooms", "grade", "Years"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, cols_hist):
        ax.hist(df[col].dropna(), bins=40, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Mapa de correlações")
    corr_vars = ["price", "sqft_living", "grade", "bathrooms", "bedrooms",
                 "view_qtde", "condition", "floors", "Years", "sqft_above"]
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[corr_vars].corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax2, annot_kws={"size": 9})
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.subheader("Distribuição geográfica dos imóveis")
    st.map(df[["lat", "long"]].rename(columns={"lat": "latitude", "long": "longitude"}))

# ════════════════════════════════════════════════════════════════════════════
# H1 – ÁREA vs PREÇO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H1 – Área vs Preço":
    st.title("H1 — Quanto maior a área construída, maior o preço do imóvel?")
    st.markdown("> **Hipótese:** Existe correlação positiva entre a área habitável (*sqft_living*) e o preço.")

    corr = df["sqft_living"].corr(df["price"])
    st.metric("Correlação de Pearson (sqft_living × price)", f"r = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H1 — Área Construída vs Preço", fontsize=13, fontweight="bold")

    axes[0].scatter(df["sqft_living"], df["price"], alpha=0.3, s=4, color="steelblue")
    coef = np.polyfit(df["sqft_living"], df["price"], 1)
    x_r = np.linspace(df["sqft_living"].min(), df["sqft_living"].max(), 200)
    axes[0].plot(x_r, np.polyval(coef, x_r), color="red", linewidth=2, label="Tendência")
    axes[0].set_title("Dispersão: Área vs Preço")
    axes[0].set_xlabel("Área Habitável (sqft_living)")
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
        axes[1].text(b.get_x() + b.get_width()/2, v + 3000, f"${v/1e3:.0f}k", ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H1 CONFIRMADA** — correlação forte: r = {corr:.4f}. Imóveis maiores tendem a ser mais caros, "
               "embora outliers mostrem que a área sozinha não explica todo o preço.")

# ════════════════════════════════════════════════════════════════════════════
# H2 – GRADE vs PREÇO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H2 – Grade vs Preço":
    st.title("H2 — Qualidade de Construção (Grade) vs Preço")
    st.markdown("> **Hipótese:** Quanto maior o *grade* (padrão de construção), maior o preço do imóvel.")

    resumo_grade = df.groupby("grade")["price"].agg(["count", "mean", "median"]).reset_index()
    resumo_grade.columns = ["Grade", "Qtd. imóveis", "Preço médio", "Preço mediano"]
    st.dataframe(resumo_grade.style.format({"Preço médio": "${:,.0f}", "Preço mediano": "${:,.0f}"}),
                 use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H2 — Grade de Construção vs Preço", fontsize=13, fontweight="bold")

    sns.barplot(data=resumo_grade, x="Grade", y="Preço médio", palette="YlOrRd", ax=axes[0])
    axes[0].set_title("Preço Médio por Grade")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("Preço Médio (USD)")
    axes[0].tick_params(axis="x", rotation=0)

    sns.boxplot(data=df, x="grade", y="price", palette="YlOrRd", ax=axes[1])
    axes[1].set_title("Distribuição de Preço por Grade")
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("Preço (USD)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success("✅ **H2 CONFIRMADA** — grade tem forte influência no preço. Quanto maior o grade, maior é "
               "o preço, visível tanto na média quanto na mediana do boxplot.")

# ════════════════════════════════════════════════════════════════════════════
# H3 – WATERFRONT vs PREÇO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H3 – Waterfront vs Preço":
    st.title("H3 — Imóveis com vista para água são mais caros?")
    st.markdown("> **Hipótese:** Imóveis com acesso à orla (*waterfront = 1*) têm preço médio significativamente maior.")

    media_wf = df.groupby("waterfront")["price"].mean()
    sem, com_ = media_wf[0], media_wf[1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Preço médio SEM orla", f"${sem:,.0f}")
    c2.metric("Preço médio COM orla", f"${com_:,.0f}")
    c3.metric("Diferença", f"+{(com_/sem - 1)*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H3 — Waterfront vs Preço", fontsize=13, fontweight="bold")

    sns.boxplot(data=df, x="waterfront", y="price", palette=["#5b9bd5", "#1f5fa6"], ax=axes[0])
    axes[0].set_xticklabels(["Sem vista para água", "Com vista para água"])
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

    st.success(f"✅ **H3 CONFIRMADA** — imóveis com orla custam {(com_/sem - 1)*100:.1f}% mais em média.")

# ════════════════════════════════════════════════════════════════════════════
# H4 – BANHEIROS vs QUARTOS
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H4 – Banheiros vs Quartos":
    st.title("H4 — Imóveis com mais banheiros geralmente têm mais quartos?")
    st.markdown("> **Hipótese:** Existe correlação positiva entre número de banheiros e número de quartos.")

    corr = df["bathrooms"].corr(df["bedrooms"])
    st.metric("Correlação de Pearson (banheiros × quartos)", f"r = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H4 — Banheiros vs Quartos", fontsize=13, fontweight="bold")

    df_plot = df[(df["bathrooms"] <= 6) & (df["bedrooms"].between(1, 8))]
    axes[0].scatter(df_plot["bathrooms"], df_plot["bedrooms"], alpha=0.3, s=8, color="coral")
    coef = np.polyfit(df_plot["bathrooms"], df_plot["bedrooms"], 1)
    x_r = np.linspace(df_plot["bathrooms"].min(), df_plot["bathrooms"].max(), 100)
    axes[0].plot(x_r, np.polyval(coef, x_r), color="red", linewidth=2, label="Tendência")
    axes[0].set_title(f"Dispersão: Banheiros vs Quartos (r = {corr:.2f})")
    axes[0].set_xlabel("Banheiros")
    axes[0].set_ylabel("Quartos")
    axes[0].legend()

    media_bath_bed = df_plot.groupby("bathrooms")["bedrooms"].mean()
    axes[1].bar(media_bath_bed.index, media_bath_bed.values, color="coral", edgecolor="white", width=0.3)
    axes[1].set_title("Média de Quartos por Número de Banheiros")
    axes[1].set_xlabel("Banheiros")
    axes[1].set_ylabel("Quartos (média)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"ℹ️ **H4 PARCIALMENTE CONFIRMADA** — correlação mediana (r = {corr:.2f}). "
            "Mais banheiros tende a acompanhar mais quartos, mas não é uma regra absoluta.")

# ════════════════════════════════════════════════════════════════════════════
# H5 – REFORMADOS SÃO MAIS ANTIGOS
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H5 – Reformados são mais antigos?":
    st.title("H5 — Imóveis reformados tendem a ser mais antigos?")
    st.markdown("> **Hipótese:** Imóveis que passaram por reforma (*yrs_renovated > 0*) tendem a ter maior idade.")

    reformados = df[df["yrs_renovated"] > 0]["Years"]
    nao_reformados = df[df["yrs_renovated"] == 0]["Years"]
    media_r = reformados.mean()
    media_nr = nao_reformados.mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Idade média – Reformados", f"{media_r:.1f} anos")
    c2.metric("Idade média – Não reformados", f"{media_nr:.1f} anos")
    c3.metric("Imóveis reformados", f"{len(reformados):,} ({len(reformados)/len(df)*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H5 — Antiguidade: Reformados vs Não Reformados", fontsize=13, fontweight="bold")

    sns.boxplot(data=[reformados, nao_reformados], showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "red"}, ax=axes[0])
    axes[0].set_xticklabels(["Reformados", "Não reformados"])
    axes[0].set_title("Distribuição de Idade")
    axes[0].set_ylabel("Idade do imóvel (anos)")

    for dados, label, cor in [(reformados, "Reformados", "#fc8d62"),
                               (nao_reformados, "Não reformados", "#66c2a5")]:
        axes[1].hist(dados, bins=40, alpha=0.55, label=label, color=cor, density=True)
    axes[1].axvline(media_r, color="#e05a20", linestyle="--", linewidth=2,
                    label=f"Média reformados: {media_r:.0f} anos")
    axes[1].axvline(media_nr, color="#3aa882", linestyle="--", linewidth=2,
                    label=f"Média não reformados: {media_nr:.0f} anos")
    axes[1].set_title("Distribuição de Idades (densidade)")
    axes[1].set_xlabel("Idade (anos)")
    axes[1].set_ylabel("Densidade")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H5 CONFIRMADA** — a idade média de imóveis reformados ({media_r:.1f} anos) "
               f"é maior que a dos não reformados ({media_nr:.1f} anos).")

# ════════════════════════════════════════════════════════════════════════════
# H6 – WATERFRONT e VISTA
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H6 – Waterfront e Vista":
    st.title("H6 — Imóveis com waterfront têm melhor vista?")
    st.markdown("> **Hipótese:** Imóveis à beira d'água (*waterfront = 1*) tendem a ter nota de vista (*view*) mais alta.")

    media_view_wf = df[df["waterfront"] == 1]["view_qtde"].mean()
    media_view_sem = df[df["waterfront"] == 0]["view_qtde"].mean()

    c1, c2 = st.columns(2)
    c1.metric("Vista média – COM waterfront", f"{media_view_wf:.2f}")
    c2.metric("Vista média – SEM waterfront", f"{media_view_sem:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H6 — Waterfront vs Vista (view)", fontsize=13, fontweight="bold")

    sns.boxplot(data=df, x="waterfront", y="view_qtde",
                palette=["#5b9bd5", "#1f5fa6"], ax=axes[0])
    axes[0].set_xticklabels(["Sem waterfront", "Com waterfront"])
    axes[0].set_title("Distribuição da Nota de Vista")
    axes[0].set_ylabel("Nota de view (0–4)")
    axes[0].set_xlabel("")

    proporcao = df.groupby("waterfront")["view_qtde"].value_counts(normalize=True).unstack(fill_value=0)
    proporcao.index = ["Sem waterfront", "Com waterfront"]
    proporcao.plot(kind="bar", stacked=True, ax=axes[1], colormap="Blues", edgecolor="white")
    axes[1].set_title("Distribuição das Notas de Vista (proporção)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Proporção")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="View", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H6 CONFIRMADA** — imóveis com waterfront têm nota de vista média de "
               f"{media_view_wf:.2f} vs {media_view_sem:.2f} dos sem waterfront.")

# ════════════════════════════════════════════════════════════════════════════
# H7 – LOCALIZAÇÃO e PADRÃO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H7 – Localização e Padrão":
    st.title("H7 — A localização influencia o padrão dos imóveis?")
    st.markdown("> **Hipótese:** A posição geográfica (lat/long) está associada ao padrão de construção (*grade*).")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H7 — Localização vs Grade", fontsize=13, fontweight="bold")

    scatter = axes[0].scatter(df["long"], df["lat"], c=df["grade"], cmap="viridis",
                               alpha=0.5, s=4)
    plt.colorbar(scatter, ax=axes[0], label="Grade")
    axes[0].set_title("Grade por Localização Geográfica")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")

    df_tmp = df.copy()
    df_tmp["lat_bin"] = pd.cut(df_tmp["lat"], bins=5)
    df_tmp["long_bin"] = pd.cut(df_tmp["long"], bins=5)
    media_grade = df_tmp.groupby(["lat_bin", "long_bin"], observed=True)["grade"].mean().unstack()
    sns.heatmap(media_grade, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Grade Médio por Região (lat × long)")
    axes[1].set_xlabel("Faixa de Longitude")
    axes[1].set_ylabel("Faixa de Latitude")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.warning("⚠️ **H7 PARCIALMENTE CONFIRMADA** — algumas regiões apresentam grade mais alto, "
               "mas a relação não é uniforme em todo o condado.")

# ════════════════════════════════════════════════════════════════════════════
# H8 – CONDIÇÃO vs GRADE
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H8 – Condição vs Grade":
    st.title("H8 — Condition alta implica Grade alto?")
    st.markdown("> **Hipótese:** Imóveis com maior *condition* **não** necessariamente possuem maior *grade*.")

    corr = df["condition"].corr(df["grade"])
    st.metric("Correlação de Pearson (condition × grade)", f"r = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H8 — Condition vs Grade", fontsize=13, fontweight="bold")

    sns.stripplot(data=df, x="condition", y="grade", jitter=True, alpha=0.3,
                  palette="Set2", ax=axes[0])
    axes[0].set_title("Distribuição do Grade por Condition")
    axes[0].set_xlabel("Condition (1–5)")
    axes[0].set_ylabel("Grade (1–13)")

    media_cond_grade = df.groupby("condition")["grade"].mean()
    axes[1].bar(media_cond_grade.index, media_cond_grade.values,
                color=sns.color_palette("Set2", len(media_cond_grade)), edgecolor="white")
    axes[1].set_title("Grade Médio por Condition")
    axes[1].set_xlabel("Condition")
    axes[1].set_ylabel("Grade médio")
    for x, v in zip(media_cond_grade.index, media_cond_grade.values):
        axes[1].text(x, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"ℹ️ **H8 CONFIRMADA** — a correlação entre condition e grade é fraca (r = {corr:.2f}). "
            "Imóveis bem conservados não são necessariamente de alto padrão construtivo.")

# ════════════════════════════════════════════════════════════════════════════
# H9 – ANDARES vs ÁREA
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H9 – Andares vs Área":
    st.title("H9 — Mais andares implica maior área construída?")
    st.markdown("> **Hipótese:** Imóveis com mais andares (*floors*) tendem a ter maior área habitável (*sqft_living*).")

    corr = df["floors"].corr(df["sqft_living"])
    st.metric("Correlação de Pearson (floors × sqft_living)", f"r = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H9 — Andares vs Área Construída", fontsize=13, fontweight="bold")

    sns.boxplot(data=df, x="floors", y="sqft_living", palette="Blues", ax=axes[0])
    axes[0].set_title("Área Habitável por Número de Andares")
    axes[0].set_xlabel("Andares (floors)")
    axes[0].set_ylabel("Área habitável (sqft_living)")

    media_floors = df.groupby("floors")["sqft_living"].mean()
    axes[1].bar([str(f) for f in media_floors.index], media_floors.values,
                color=sns.color_palette("Blues", len(media_floors)), edgecolor="white")
    axes[1].set_title("Área Média por Número de Andares")
    axes[1].set_xlabel("Andares")
    axes[1].set_ylabel("Área média (sqft_living)")
    for i, (f, v) in enumerate(zip(media_floors.index, media_floors.values)):
        axes[1].text(i, v + 20, f"{v:,.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info(f"ℹ️ **H9 PARCIALMENTE CONFIRMADA** — mais andares tende a correlacionar com mais área "
            f"(r = {corr:.2f}), mas a relação não é totalmente linear.")

# ════════════════════════════════════════════════════════════════════════════
# H10 – LOTE vs ÁREA CONSTRUÍDA
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "H10 – Lote vs Área Construída":
    st.title("H10 — Lotes maiores têm maior área construída?")
    st.markdown("> **Hipótese:** Imóveis com maior terreno (*sqft_lot*) **não** necessariamente possuem "
                "maior área construída (*sqft_living*).")

    corr = df["sqft_lot"].corr(df["sqft_living"])
    st.metric("Correlação de Pearson (sqft_lot × sqft_living)", f"r = {corr:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("H10 — Lote vs Área Construída", fontsize=13, fontweight="bold")

    df_plot = df[df["sqft_lot"] < df["sqft_lot"].quantile(0.95)]
    axes[0].scatter(df_plot["sqft_lot"], df_plot["sqft_living"], alpha=0.3, s=4, color="teal")
    coef = np.polyfit(df_plot["sqft_lot"], df_plot["sqft_living"], 1)
    x_r = np.linspace(df_plot["sqft_lot"].min(), df_plot["sqft_lot"].max(), 200)
    axes[0].plot(x_r, np.polyval(coef, x_r), color="red", linewidth=2, label="Tendência")
    axes[0].set_title(f"Dispersão: Lote vs Área Construída (r = {corr:.2f})")
    axes[0].set_xlabel("Área do Lote (sqft_lot)")
    axes[0].set_ylabel("Área Construída (sqft_living)")
    axes[0].legend()

    faixas = pd.cut(df_plot["sqft_lot"], bins=6)
    media_por_faixa = df_plot.groupby(faixas, observed=True)["sqft_living"].mean()
    faixa_labels = [f"{int(i.left/1000)}k–{int(i.right/1000)}k" for i in media_por_faixa.index]
    axes[1].bar(range(len(faixa_labels)), media_por_faixa.values,
                color="teal", edgecolor="white", alpha=0.8)
    axes[1].set_xticks(range(len(faixa_labels)))
    axes[1].set_xticklabels(faixa_labels, rotation=20, fontsize=8)
    axes[1].set_title("Área Construída Média por Faixa de Lote")
    axes[1].set_xlabel("Faixa de Lote")
    axes[1].set_ylabel("Área média (sqft_living)")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(f"✅ **H10 CONFIRMADA** — correlação muito fraca (r = {corr:.2f}). "
               "Lotes grandes não garantem casas grandes; os dois são independentes.")

# ════════════════════════════════════════════════════════════════════════════
# PREVISÃO DE PREÇO
# ════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Previsão de Preço":
    st.title("🤖 Previsão de Preço — Random Forest")
    st.markdown(
        "Modelo treinado com os dados tratados de King County. "
        "Ajuste as características do imóvel e veja o preço estimado."
    )

    st.subheader("Desempenho do modelo (conjunto de teste — 20%)")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{metrics['r2']:.4f}", help="Quanto maior, melhor (máx = 1)")
    c2.metric("MAE", f"${metrics['mae']:,.0f}", help="Erro médio absoluto")
    c3.metric("RMSE", f"${metrics['rmse']:,.0f}", help="Raiz do erro quadrático médio")

    with st.expander("📊 Ver importância das variáveis"):
        fig_imp, ax_imp = plt.subplots(figsize=(9, 4))
        importances.plot(kind="bar", ax=ax_imp, color="steelblue", edgecolor="white")
        ax_imp.set_title("Importância das Features (Random Forest)")
        ax_imp.set_ylabel("Importância")
        ax_imp.tick_params(axis="x", rotation=35)
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

    st.markdown("---")
    st.subheader("Configure o imóvel")

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
        view_val    = st.slider("Nota de vista (0–4)", 0, 4, 0)
        lat         = st.number_input("Latitude", value=47.56, min_value=47.15, max_value=47.78, step=0.01)
        long        = st.number_input("Longitude", value=-122.21, min_value=-122.52, max_value=-121.31, step=0.01)

    waterfront_val = 1 if waterfront == "Sim" else 0
    years_val = 2026 - yr_built

    entrada = np.array([[sqft_living, grade, bathrooms, bedrooms, floors,
                         waterfront_val, view_val, condition, sqft_above, years_val, lat, long]])
    entrada_scaled = scaler.transform(entrada)
    preco_previsto = model.predict(entrada_scaled)[0]

    st.markdown("---")
    st.markdown(
        f"<h2 style='text-align:center; color:#1f5fa6;'>💰 Preço estimado: "
        f"<strong>${preco_previsto:,.0f}</strong></h2>",
        unsafe_allow_html=True,
    )

    media_geral = df["price"].mean()
    diff_pct = (preco_previsto / media_geral - 1) * 100
    simbolo = "acima" if diff_pct >= 0 else "abaixo"
    st.caption(f"Preço médio do dataset: ${media_geral:,.0f} — "
               f"estimativa {abs(diff_pct):.1f}% {simbolo} da média.")
