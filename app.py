import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os


# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Mortalidade Infantil no Pará",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo visual leve e seguro
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }

        h1 {
            font-size: 2.6rem;
            font-weight: 700;
        }

        h2, h3 {
            font-weight: 650;
        }

        .stMetric {
            padding: 10px;
            border-radius: 10px;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
        }

        section[data-testid="stSidebar"] {
            padding-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

municipios_para = {
    "150125": "Bannach",
    "150010": "Abaetetuba",
    "150013": "Abel Figueiredo",
    "150020": "Acará",
    "150030": "Afuá",
    "150034": "Água Azul do Norte",
    "150040": "Alenquer",
    "150050": "Almeirim",
    "150060": "Altamira",
    "150070": "Anajás",
    "150080": "Ananindeua",
    "150085": "Anapu",
    "150090": "Augusto Corrêa",
    "150095": "Aurora do Pará",
    "150100": "Aveiro",
    "150110": "Bagre",
    "150120": "Baião",
    "150130": "Barcarena",
    "150140": "Belém",
    "150145": "Belterra",
    "150150": "Benevides",
    "150157": "Bom Jesus do Tocantins",
    "150160": "Bonito",
    "150170": "Bragança",
    "150172": "Brasil Novo",
    "150175": "Brejo Grande do Araguaia",
    "150178": "Breu Branco",
    "150180": "Breves",
    "150190": "Bujaru",
    "150195": "Cachoeira do Piriá",
    "150200": "Cachoeira do Arari",
    "150210": "Cametá",
    "150215": "Canaã dos Carajás",
    "150220": "Capanema",
    "150230": "Capitão Poço",
    "150240": "Castanhal",
    "150250": "Chaves",
    "150260": "Colares",
    "150270": "Conceição do Araguaia",
    "150275": "Concórdia do Pará",
    "150276": "Cumaru do Norte",
    "150277": "Curionópolis",
    "150280": "Curralinho",
    "150285": "Curuá",
    "150290": "Curuçá",
    "150293": "Dom Eliseu",
    "150295": "Eldorado do Carajás",
    "150300": "Faro",
    "150304": "Floresta do Araguaia",
    "150307": "Garrafão do Norte",
    "150309": "Goianésia do Pará",
    "150310": "Gurupá",
    "150320": "Igarapé-Açu",
    "150330": "Igarapé-Miri",
    "150340": "Inhangapi",
    "150345": "Ipixuna do Pará",
    "150350": "Irituia",
    "150360": "Itaituba",
    "150370": "Itupiranga",
    "150375": "Jacareacanga",
    "150380": "Jacundá",
    "150390": "Juruti",
    "150400": "Limoeiro do Ajuru",
    "150405": "Mãe do Rio",
    "150410": "Magalhães Barata",
    "150420": "Marabá",
    "150430": "Maracanã",
    "150440": "Marapanim",
    "150442": "Marituba",
    "150445": "Medicilândia",
    "150450": "Melgaço",
    "150460": "Mocajuba",
    "150470": "Moju",
    "150475": "Mojuí dos Campos",
    "150480": "Monte Alegre",
    "150490": "Muaná",
    "150495": "Nova Esperança do Piriá",
    "150497": "Nova Ipixuna",
    "150500": "Nova Timboteua",
    "150503": "Novo Progresso",
    "150506": "Novo Repartimento",
    "150510": "Óbidos",
    "150520": "Oeiras do Pará",
    "150530": "Oriximiná",
    "150540": "Ourém",
    "150543": "Ourilândia do Norte",
    "150548": "Pacajá",
    "150549": "Palestina do Pará",
    "150550": "Paragominas",
    "150553": "Parauapebas",
    "150555": "Pau d'Arco",
    "150560": "Peixe-Boi",
    "150563": "Piçarra",
    "150565": "Placas",
    "150570": "Ponta de Pedras",
    "150580": "Portel",
    "150590": "Porto de Moz",
    "150600": "Prainha",
    "150610": "Primavera",
    "150611": "Quatipuru",
    "150613": "Redenção",
    "150616": "Rio Maria",
    "150618": "Rondon do Pará",
    "150619": "Rurópolis",
    "150620": "Salinópolis",
    "150630": "Salvaterra",
    "150635": "Santa Bárbara do Pará",
    "150640": "Santa Cruz do Arari",
    "150650": "Santa Isabel do Pará",
    "150655": "Santa Luzia do Pará",
    "150658": "Santa Maria das Barreiras",
    "150660": "Santa Maria do Pará",
    "150670": "Santana do Araguaia",
    "150680": "Santarém",
    "150690": "Santarém Novo",
    "150700": "Santo Antônio do Tauá",
    "150710": "São Caetano de Odivelas",
    "150715": "São Domingos do Araguaia",
    "150720": "São Domingos do Capim",
    "150730": "São Félix do Xingu",
    "150740": "São Francisco do Pará",
    "150745": "São Geraldo do Araguaia",
    "150746": "São João da Ponta",
    "150747": "São João de Pirabas",
    "150750": "São João do Araguaia",
    "150760": "São Miguel do Guamá",
    "150770": "São Sebastião da Boa Vista",
    "150775": "Sapucaia",
    "150780": "Senador José Porfírio",
    "150790": "Soure",
    "150795": "Tailândia",
    "150796": "Terra Alta",
    "150797": "Terra Santa",
    "150800": "Tomé-Açu",
    "150803": "Tracuateua",
    "150805": "Trairão",
    "150808": "Tucumã",
    "150810": "Tucuruí",
    "150812": "Ulianópolis",
    "150815": "Uruará",
    "150820": "Vigia",
    "150830": "Viseu",
    "150835": "Vitória do Xingu",
    "150840": "Xinguara",
    "ignorado": "Ignorado",
    "referencia": "Município de referência"
}

# =====================================================
# CARREGAMENTO DOS DADOS E MODELO
# =====================================================

@st.cache_data
def carregar_dados():
    """
    Carrega a base completa se ela estiver disponível.
    Caso contrário, utiliza a amostra de 50 mil registros.
    """
    import os

    caminho_parquet = "data/base_modelagem.parquet"
    caminho_csv_completo = "data/base_modelagem.csv"
    caminho_csv_amostra = "data/base_modelagem_amostra.csv"

    if os.path.exists(caminho_parquet):
        return pd.read_parquet(caminho_parquet), "Base completa em Parquet"

    elif os.path.exists(caminho_csv_completo):
        return pd.read_csv(caminho_csv_completo), "Base completa em CSV"

    else:
        return pd.read_csv(caminho_csv_amostra), "Amostra de 50.000 registros"


@st.cache_resource
def carregar_modelo():
    modelo = XGBClassifier()
    modelo.load_model("models/modelo_xgboost_v2.json")

    colunas_modelo = joblib.load("models/colunas_modelo.pkl")

    return modelo, colunas_modelo


df, tipo_base = carregar_dados()
modelo, colunas_modelo = carregar_modelo()

# =====================================================
# FUNÇÕES AUXILIARES
# =====================================================

def formatar_numero(valor):
    return f"{valor:,}".replace(",", ".")


def criar_entrada_modelo(
    ano_nasc,
    mes_nasc,
    municipio,
    sexo,
    peso,
    idade_mae,
    escolaridade_mae,
    filhos_vivos,
    filhos_mortos,
    gestacao,
    gravidez,
    parto
):
    """
    Cria uma linha de entrada compatível com as colunas usadas no treinamento.
    As variáveis categóricas foram codificadas com get_dummies.
    Categorias de referência ficam com todas as colunas zeradas.
    """

    entrada = pd.DataFrame(0, index=[0], columns=colunas_modelo)

    if "ano_nasc" in entrada.columns:
        entrada.loc[0, "ano_nasc"] = ano_nasc

    if "mes_nasc" in entrada.columns:
        entrada.loc[0, "mes_nasc"] = mes_nasc

    colunas_categoricas = [
        f"CODMUNRES_{municipio}",
        f"sexo_cat_{sexo}",
        f"peso_cat_{peso}",
        f"idade_mae_cat_{idade_mae}",
        f"esc_mae_cat_{escolaridade_mae}",
        f"filhos_vivos_cat_{filhos_vivos}",
        f"filhos_mortos_cat_{filhos_mortos}",
        f"gestacao_cat_{gestacao}",
        f"gravidez_cat_{gravidez}",
        f"parto_cat_{parto}"
    ]

    for coluna in colunas_categoricas:
        if coluna in entrada.columns:
            entrada.loc[0, coluna] = 1

    return entrada


def classificar_probabilidade(probabilidade_percentual):
    if probabilidade_percentual < 5:
        return "Baixa probabilidade estimada"
    elif probabilidade_percentual < 15:
        return "Probabilidade estimada moderada"
    else:
        return "Probabilidade estimada elevada"


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Projeto NAP2")

st.sidebar.markdown(
    """
    **Tema:** Mortalidade Infantil no Pará  
    **Bases:** SIM e SINASC  
    **Modelo:** XGBoost  
    **Interpretação:** SHAP  
    **Deploy:** Streamlit
    """
)

pagina = st.sidebar.radio(
    "Navegação",
    [
        "Visão geral",
        "Modelo e avaliação",
        "Fatores associados",
        "Causas de óbito",
        "Análise territorial",
        "Simulador de risco",
        "Arquitetura e MLOps",
        "Ética e limitações"
    ]
)


# =====================================================
# TÍTULO GERAL
# =====================================================

st.title("Análise da Mortalidade Infantil no Estado do Pará")

st.markdown(
    """
    Dashboard desenvolvido como parte da **Entrega NAP2** do projeto de Ciência de Dados,
    utilizando dados públicos dos sistemas **SIM** e **SINASC**, referentes ao estado do Pará.

    O objetivo é apoiar a análise da mortalidade infantil por meio de indicadores,
    visualizações, modelagem preditiva com **XGBoost** e interpretação dos fatores associados.
    """
)

st.divider()


# =====================================================
# PÁGINA 1 — VISÃO GERAL
# =====================================================

if pagina == "Visão geral":

    st.info(f"Base carregada: {tipo_base}")

    st.header("1. Visão geral da base")

    st.markdown(
        """
        Esta seção apresenta uma visão geral da base integrada utilizada no projeto, 
        incluindo o total de registros analisados, a quantidade de óbitos infantis associados,
        o percentual de óbitos e a taxa de mortalidade infantil por mil nascidos vivos.
        """
    )

    # =========================
    # Filtro por ano
    # =========================

    anos_disponiveis = sorted(df["ano_nasc"].dropna().unique())

    anos_selecionados = st.multiselect(
        "Filtrar por ano de nascimento",
        options=anos_disponiveis,
        default=anos_disponiveis
    )

    df_filtrado = df[df["ano_nasc"].isin(anos_selecionados)]

    # =========================
    # Indicadores gerais
    # =========================

    total_registros = len(df_filtrado)
    total_obitos = int(df_filtrado["obito_infantil"].sum())
    total_nao_obitos = total_registros - total_obitos

    if total_registros > 0:
        percentual_obitos = (total_obitos / total_registros) * 100
        taxa_por_mil = (total_obitos / total_registros) * 1000
    else:
        percentual_obitos = 0
        taxa_por_mil = 0

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Registros analisados", formatar_numero(total_registros))
    col2.metric("Sem óbito infantil", formatar_numero(total_nao_obitos))
    col3.metric("Com óbito infantil", formatar_numero(total_obitos))
    col4.metric("Percentual de óbitos", f"{percentual_obitos:.2f}%".replace(".", ","))
    col5.metric("Taxa por mil", f"{taxa_por_mil:.2f}".replace(".", ","))

    st.markdown(
        """
        A variável `obito_infantil` indica se o registro de nascimento foi associado ou não
        a um óbito infantil. Como os óbitos representam uma pequena parcela em relação ao total
        de nascidos vivos, a base apresenta forte desbalanceamento entre as classes.

        A taxa por mil representa a quantidade de óbitos infantis associados a cada mil nascidos vivos
        registrados na base integrada.
        """
    )

    # =========================
    # Distribuição da variável-alvo
    # =========================

    st.subheader("Distribuição da variável-alvo")

    dist = pd.DataFrame({
        "Classe": ["Não óbito", "Óbito infantil"],
        "Quantidade": [total_nao_obitos, total_obitos]
    })

    st.dataframe(dist, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.bar(dist["Classe"], dist["Quantidade"])
    ax.set_title("Distribuição da variável-alvo", fontsize=11)
    ax.set_ylabel("Quantidade", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)

    # Remove notação científica tipo 1e6
    ax.ticklabel_format(style="plain", axis="y")

    for i, valor in enumerate(dist["Quantidade"]):
        ax.text(
            i,
            valor,
            formatar_numero(valor),
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()

    col_grafico, col_texto = st.columns([1, 1])

    with col_grafico:
        st.pyplot(fig)

    with col_texto:
        st.markdown(
            """
            A distribuição evidencia o forte desbalanceamento da base. 
            A quantidade de registros sem óbito infantil associado é muito superior 
            à quantidade de registros com óbito infantil.

            Esse comportamento justifica o uso de métricas além da acurácia na avaliação do modelo.
            """
        )

    # =========================
    # Série histórica
    # =========================

    st.subheader("Série histórica da mortalidade infantil")

    if total_registros > 0:

        serie = df_filtrado.groupby("ano_nasc").agg(
            nascimentos=("obito_infantil", "count"),
            obitos=("obito_infantil", "sum")
        ).reset_index()

        serie["taxa_por_mil"] = (serie["obitos"] / serie["nascimentos"]) * 1000

        serie_exibicao = serie.copy()
        serie_exibicao["taxa_por_mil"] = serie_exibicao["taxa_por_mil"].round(2)

        serie_exibicao = serie_exibicao.rename(columns={
            "ano_nasc": "Ano",
            "nascimentos": "Nascidos vivos",
            "obitos": "Óbitos infantis",
            "taxa_por_mil": "Taxa por mil"
        })

        st.dataframe(serie_exibicao, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(
            serie["ano_nasc"],
            serie["taxa_por_mil"],
            marker="o"
        )

        ax2.set_title("Taxa de mortalidade infantil por ano")
        ax2.set_xlabel("Ano de nascimento")
        ax2.set_ylabel("Óbitos por mil nascidos vivos")
        ax2.grid(True, alpha=0.3)

        for _, row in serie.iterrows():
            ax2.text(
                row["ano_nasc"],
                row["taxa_por_mil"],
                f'{row["taxa_por_mil"]:.2f}'.replace(".", ","),
                ha="center",
                va="bottom",
                fontsize=8
            )

        st.pyplot(fig2)

        st.markdown(
            """
            A série histórica permite observar a evolução da taxa de mortalidade infantil ao longo dos anos,
            considerando os registros disponíveis na base integrada. Essa análise é importante para identificar
            variações temporais, possíveis inconsistências e tendências gerais do fenômeno.
            """
        )

    else:
        st.warning("Nenhum registro encontrado para o filtro selecionado.")

    # =========================
    # Prévia da base
    # =========================

    st.subheader("Prévia da base filtrada")

    st.dataframe(df_filtrado.head(20), use_container_width=True)


# =====================================================
# PÁGINA 2 — MODELO E AVALIAÇÃO
# =====================================================

elif pagina == "Modelo e avaliação":

    st.header("2. Modelo preditivo e avaliação")

    st.markdown(
        """
        Esta seção apresenta o modelo final adotado no projeto e os principais resultados
        da avaliação preditiva. O objetivo do modelo é classificar a variável-alvo
        `obito_infantil`, indicando se determinado registro de nascimento está ou não
        associado à ocorrência de óbito infantil.
        """
    )

    # =========================
    # Modelo final
    # =========================

    st.subheader("Modelo final adotado")

    st.markdown(
        """
        O modelo final escolhido foi o **XGBoost V2**, utilizado para uma tarefa de
        **classificação binária**. A escolha do XGBoost se justifica por sua capacidade
        de trabalhar com dados estruturados, lidar com relações não lineares entre as
        variáveis e apresentar bom desempenho em problemas com bases desbalanceadas.

        Para fins de deploy, o modelo foi salvo em formato nativo do XGBoost
        (`modelo_xgboost_v2.json`) e integrado ao dashboard em Streamlit.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Modelo final", "XGBoost V2")
    col2.metric("Tarefa", "Classificação binária")
    col3.metric("Variável-alvo", "obito_infantil")
    col4.metric("AUC-ROC", "0,80")

    st.info(
        """
        A AUC-ROC de 0,80 indica que o modelo apresentou capacidade discriminativa satisfatória
        para diferenciar registros associados e não associados ao óbito infantil.
        """
    )

    # =========================
    # Métricas
    # =========================

    st.subheader("Métricas de avaliação")

    metricas = pd.DataFrame({
        "Métrica": [
            "Acurácia",
            "Precision — classe óbito infantil",
            "Recall — classe óbito infantil",
            "F1-score — classe óbito infantil",
            "AUC-ROC",
            "Average Precision"
        ],
        "Valor": [
            "0,841",
            "0,054",
            "0,642",
            "0,099",
            "0,80",
            "0,101"
        ],
        "Interpretação": [
            "Proporção geral de acertos do modelo.",
            "Entre os casos classificados como óbito infantil, indica quantos realmente eram óbito.",
            "Entre os óbitos infantis reais, indica quantos foram identificados pelo modelo.",
            "Combina precision e recall em uma única medida.",
            "Mede a capacidade geral de separação entre as classes.",
            "Resume o desempenho na curva Precision-Recall."
        ]
    })

    st.dataframe(metricas, use_container_width=True)

    st.markdown(
        """
        Como a base é fortemente desbalanceada, a acurácia isoladamente não é suficiente
        para avaliar o desempenho do modelo. Por isso, foram consideradas métricas como
        **precision**, **recall**, **F1-score**, **AUC-ROC** e **Average Precision**.
        """
    )

    # =========================
    # Matriz de confusão
    # =========================

    matriz = pd.DataFrame({
        "Classe real": ["Real: Não óbito", "Real: Óbito infantil"],
        "Predito: Não óbito": [223117, 1299],
        "Predito: Óbito infantil": [41182, 2331]
    })

    st.dataframe(matriz, use_container_width=True, hide_index=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Verdadeiros negativos", "223.117")
    col2.metric("Falsos positivos", "41.182")
    col3.metric("Falsos negativos", "1.299")
    col4.metric("Verdadeiros positivos", "2.331")

    st.markdown(
    """
    A matriz de confusão mostra que o modelo identificou parte relevante dos registros
    associados ao óbito infantil, porém também gerou um número considerável de falsos positivos.
    Esse comportamento está relacionado ao forte desbalanceamento da base, em que os registros
    de óbito infantil representam uma parcela pequena em relação ao total de nascidos vivos.

    Por esse motivo, os resultados devem ser interpretados como apoio analítico, e não como
    classificação definitiva de casos individuais.
    """
    )

    # =========================
    # Interpretação crítica
    # =========================

    st.subheader("Interpretação crítica da avaliação")

    st.markdown(
        """
        O **recall da classe óbito** foi uma métrica importante neste projeto, pois indica
        a capacidade do modelo de identificar registros realmente associados ao óbito infantil.
        Em problemas de saúde pública, reduzir a perda de casos potencialmente relevantes é
        importante para apoiar análises de vigilância e planejamento.

        Por outro lado, a **precision baixa** indica que muitos registros classificados como
        possíveis óbitos pelo modelo não correspondem, de fato, à classe óbito. Isso evidencia
        a dificuldade do problema e reforça que o modelo não deve ser utilizado como ferramenta
        de decisão individual ou diagnóstico.

        Dessa forma, o modelo final deve ser interpretado como uma ferramenta complementar de
        apoio analítico, útil para identificar padrões e fatores associados, mas sempre em
        conjunto com análise epidemiológica e avaliação técnica.
        """
    )

    st.success(
    """
    Modelo final carregado, documentado e integrado ao sistema. Esta etapa atende à exigência
    de apresentação do modelo final, avaliação avançada e análise crítica dos resultados.
    """
    )


# =====================================================
# PÁGINA 3 — FATORES ASSOCIADOS / SHAP
# =====================================================

elif pagina == "Fatores associados":

    st.header("3. Fatores associados e interpretabilidade com SHAP")

    st.markdown(
        """
        Esta seção apresenta a interpretação do modelo final por meio da técnica 
        **SHAP (SHapley Additive exPlanations)**. O objetivo do SHAP é indicar quais variáveis
        tiveram maior influência nas predições do modelo, contribuindo para tornar os resultados
        mais compreensíveis e interpretáveis.
        """
    )

    # =========================
    # Explicação do SHAP
    # =========================

    st.subheader("O que é SHAP?")

    st.markdown(
        """
        O SHAP é uma técnica de interpretabilidade utilizada para explicar modelos de aprendizado
        de máquina. No contexto deste projeto, ele foi aplicado para identificar quais características
        dos registros de nascimento tiveram maior peso nas predições associadas ao óbito infantil.

        Essa etapa é importante porque, em aplicações de saúde pública, não basta apenas obter uma
        predição: também é necessário compreender quais fatores influenciaram o comportamento do modelo.
        """
    )

    st.info(
        """
        A interpretabilidade permite aproximar o resultado computacional do contexto epidemiológico,
        apoiando uma análise mais crítica e transparente do modelo.
        """
    )

    # =========================
    # Gráfico SHAP real
    # =========================

    st.subheader("Gráfico de importância das variáveis segundo SHAP")

    caminho_shap = "outputs\Shap.png"

    if os.path.exists(caminho_shap):
        col_img, col_texto = st.columns([0.75, 0.25])

        with col_img:
            st.image(
                caminho_shap,
                caption="Resumo SHAP das variáveis mais influentes no modelo XGBoost V2",
                width=650
            )
            st.markdown(
                """
                O gráfico SHAP apresenta as variáveis mais influentes nas predições do modelo.
                Cada ponto representa um registro da base analisada. No eixo horizontal, valores SHAP
                positivos indicam aumento da probabilidade estimada de associação ao óbito infantil,
                enquanto valores negativos indicam redução dessa probabilidade.

                As cores representam o valor da variável: tons mais próximos do vermelho indicam valores
                mais altos da característica, enquanto tons mais próximos do azul indicam valores mais baixos.
                Assim, o gráfico permite observar não apenas quais variáveis foram mais importantes,
                mas também como elas contribuíram para o comportamento do modelo.
                """
            )
    else:
        st.warning(
            """
            O gráfico SHAP ainda não foi encontrado na pasta `outputs/`.
            Quando o arquivo `shap_importance.png` for exportado do notebook de modelagem,
            ele será exibido automaticamente nesta seção.
            """
        )

    # =========================
    # Fatores principais
    # =========================

    st.subheader("Principais fatores associados identificados")

    fatores = pd.DataFrame({
        "Fator associado": [
            "Peso ao nascer",
            "Duração da gestação",
            "Sexo do recém-nascido",
            "Idade materna",
            "Município de residência"
        ],
        "Dimensão": [
            "Biológica",
            "Gestacional",
            "Biológica",
            "Materna",
            "Territorial"
        ],
        "Interpretação no projeto": [
            "Relaciona-se à vulnerabilidade do recém-nascido, especialmente em casos de baixo peso ou peso insuficiente.",
            "Indica a importância da idade gestacional, especialmente em situações de prematuridade ou gestação fora do padrão esperado.",
            "Pode refletir diferenças biológicas observadas na distribuição dos registros e nas predições do modelo.",
            "Representa fatores maternos associados à gestação, ao nascimento e a possíveis condições de vulnerabilidade.",
            "Evidencia a dimensão territorial da mortalidade infantil, considerando desigualdades regionais e diferenças de acesso aos serviços de saúde."
        ]
    })

    st.dataframe(fatores, use_container_width=True, hide_index=True)

    st.markdown(
        """
        Os resultados indicam que a mortalidade infantil está associada a um conjunto de fatores
        biológicos, maternos e territoriais. Isso reforça que o fenômeno não deve ser analisado
        a partir de uma única variável isolada, mas sim como resultado de múltiplas condições
        relacionadas ao nascimento, à mãe, à gestação e ao território.
        """
    )

    # =========================
    # Destaques interpretativos
    # =========================

    st.subheader("Leitura dos achados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Fatores biológicos e gestacionais**

            - Peso ao nascer;
            - Duração da gestação;
            - Sexo do recém-nascido.

            Esses fatores estão diretamente relacionados às condições do recém-nascido e ao período
            gestacional, sendo importantes para compreender vulnerabilidades no início da vida.
            """
        )

    with col2:
        st.markdown(
            """
            **Fatores maternos e territoriais**

            - Idade materna;
            - Município de residência.

            Esses fatores ampliam a análise para além do recém-nascido, incluindo aspectos ligados
            à mãe e ao território, como desigualdades regionais e acesso aos serviços de saúde.
            """
        )

    # =========================
    # Limitação da interpretação
    # =========================

    st.subheader("Cuidados na interpretação")

    st.markdown(
        """
        A interpretação por SHAP indica a influência das variáveis nas predições do modelo, mas não
        deve ser entendida automaticamente como relação causal. Ou seja, o fato de uma variável ter
        grande influência no modelo não significa, isoladamente, que ela cause o óbito infantil.

        Por isso, os resultados devem ser analisados em conjunto com o conhecimento epidemiológico,
        a literatura científica e o contexto social e territorial do estado do Pará.
        """
    )

    st.success(
        """
        A página apresenta a etapa de explainability do projeto, atendendo ao requisito de 
        interpretação do modelo e análise dos fatores associados.
        """
    )
# =====================================================
# PÁGINA 4 — CAUSAS DE ÓBITO
# =====================================================

elif pagina == "Causas de óbito":

    st.header("4. Principais causas de óbito infantil")

    st.markdown(
        """
        Esta seção apresenta a distribuição das principais causas de óbito infantil segundo
        agrupamentos da CID-10. A análise das causas complementa a modelagem preditiva,
        pois permite compreender quais grupos de condições aparecem com maior frequência
        entre os registros associados ao óbito infantil.
        """
    )

    # =========================
    # Filtrar óbitos infantis
    # =========================

    df_obitos = df[df["obito_infantil"] == 1].copy()
    total_obitos = len(df_obitos)

    col1, col2, col3 = st.columns(3)

    col1.metric("Óbitos infantis analisados", formatar_numero(total_obitos))
    col2.metric("Base carregada", tipo_base)
    col3.metric("Classificação", "CID-10")

    st.markdown(
        """
        A análise considera apenas os registros marcados com `obito_infantil = 1`,
        ou seja, registros da base integrada associados à ocorrência de óbito infantil.
        """
    )

    # =========================
    # Montagem das causas
    # =========================

    if "cid_capitulo" in df_obitos.columns:

        causas = (
            df_obitos["cid_capitulo"]
            .fillna("Ignorado")
            .value_counts()
            .reset_index()
        )

        causas.columns = ["Causa de óbito segundo a CID-10", "Quantidade"]
        causas["Percentual"] = (causas["Quantidade"] / causas["Quantidade"].sum()) * 100

        fonte_causas = "Valores calculados a partir da coluna `cid_capitulo` da base integrada."

    elif "CAUSABAS" in df_obitos.columns:

        st.info(
            """
            A base possui a coluna `CAUSABAS`, mas não possui o agrupamento por capítulo da CID-10.
            Por isso, a página apresenta os percentuais consolidados da análise exploratória anterior.
            """
        )

        causas = pd.DataFrame({
            "Causa de óbito segundo a CID-10": [
                "Afecções originadas no período perinatal",
                "Malformações congênitas",
                "Doenças do aparelho respiratório",
                "Algumas doenças infecciosas e parasitárias",
                "Demais causas"
            ],
            "Quantidade": [
                10804,
                3573,
                1143,
                962,
                1668
            ],
            "Percentual": [
                59.5,
                19.7,
                6.3,
                5.3,
                9.2
            ]
        })

        fonte_causas = "Percentuais consolidados a partir da análise exploratória dos óbitos infantis."

    else:

        causas = pd.DataFrame({
            "Causa de óbito segundo a CID-10": [
                "Afecções originadas no período perinatal",
                "Malformações congênitas",
                "Doenças do aparelho respiratório",
                "Algumas doenças infecciosas e parasitárias",
                "Demais causas"
            ],
            "Quantidade": [
                10804,
                3573,
                1143,
                962,
                1668
            ],
            "Percentual": [
                59.5,
                19.7,
                6.3,
                5.3,
                9.2
            ]
        })

        fonte_causas = (
            "Percentuais consolidados a partir da análise exploratória anterior, "
            "pois a base carregada no dashboard não contém colunas de causa de óbito agrupadas."
        )

    causas = causas.sort_values("Percentual", ascending=False)

    # =========================
    # Tabela
    # =========================

    st.subheader("Distribuição das causas de óbito")

    causas_exibicao = causas.copy()
    causas_exibicao["Percentual"] = causas_exibicao["Percentual"].round(2)

    st.dataframe(
        causas_exibicao,
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Fonte dos dados exibidos: {fonte_causas}")

    # =========================
    # Indicadores das duas principais causas
    # =========================

    st.subheader("Destaques principais")

    causa_1 = causas.iloc[0]
    causa_2 = causas.iloc[1]

    col1, col2 = st.columns(2)

    col1.metric(
        causa_1["Causa de óbito segundo a CID-10"],
        f'{causa_1["Percentual"]:.1f}%'.replace(".", ",")
    )

    col2.metric(
        causa_2["Causa de óbito segundo a CID-10"],
        f'{causa_2["Percentual"]:.1f}%'.replace(".", ",")
    )

    # =========================
    # Gráfico
    # =========================

    st.subheader("Gráfico das principais causas")

    top_causas = causas.head(5).copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.barh(
        top_causas["Causa de óbito segundo a CID-10"],
        top_causas["Percentual"]
    )

    ax.set_xlabel("Percentual dos óbitos infantis (%)", fontsize=9)
    ax.set_title("Principais causas de óbito infantil", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()

    for i, valor in enumerate(top_causas["Percentual"]):
        ax.text(
            valor + 0.5,
            i,
            f"{valor:.1f}%".replace(".", ","),
            va="center",
            fontsize=8
        )

    plt.tight_layout()

    col_grafico, col_texto = st.columns([1.1, 1])

    with col_grafico:
        st.pyplot(fig, use_container_width=False)

    with col_texto:
        st.markdown(
            """
            As causas de óbito ajudam a interpretar o fenômeno para além da predição do modelo.

            No projeto, destacaram-se as **afecções originadas no período perinatal** e as
            **malformações congênitas**, indicando a importância da atenção à gestação,
            ao parto, ao nascimento e aos primeiros dias de vida.
            """
        )

    # =========================
    # Interpretação epidemiológica
    # =========================

    st.subheader("Interpretação epidemiológica")

    st.markdown(
        """
        O predomínio das afecções originadas no período perinatal reforça a necessidade de
        atenção à gestação, ao parto, ao cuidado neonatal e ao acompanhamento dos primeiros
        dias de vida.

        A presença das malformações congênitas entre as principais causas também indica a
        importância do acompanhamento pré-natal, do diagnóstico precoce e da estrutura
        assistencial disponível para o recém-nascido.

        Dessa forma, a análise das causas complementa os resultados do XGBoost e do SHAP,
        pois ajuda a contextualizar os fatores associados à mortalidade infantil em uma
        perspectiva epidemiológica.
        """
    )

    # =========================
    # Cuidados e limitações
    # =========================

    st.subheader("Cuidados na interpretação")

    st.markdown(
        """
        As causas de óbito foram analisadas a partir dos registros disponíveis nos sistemas
        de informação em saúde. Por se tratar de dados administrativos, podem existir limitações
        relacionadas à completude, preenchimento, classificação da causa básica e qualidade
        das informações registradas.

        Por isso, os resultados devem ser interpretados como apoio à análise epidemiológica
        e não como avaliação definitiva da qualidade da assistência em cada localidade.
        """
    )

    st.success(
        """
        A página de causas de óbito complementa a análise preditiva com uma leitura epidemiológica
        dos principais grupos de causas associados aos óbitos infantis.
        """
    )
# =====================================================
# PÁGINA 5 — ANÁLISE TERRITORIAL
# =====================================================

elif pagina == "Análise territorial":

    st.header("5. Análise territorial da mortalidade infantil")

    st.markdown(
        """
        Esta seção apresenta uma análise territorial dos registros da base integrada,
        considerando o município de residência. A dimensão territorial é importante porque
        a mortalidade infantil pode estar relacionada a desigualdades regionais, diferenças
        de acesso aos serviços de saúde e condições socioeconômicas distintas entre os municípios.
        """
    )

    # =========================
    # Identificação da coluna de município
    # =========================

    if "CODMUNRES" in df.columns:

        df_municipios = df.copy()
        df_municipios["municipio_cod"] = df_municipios["CODMUNRES"].astype(str)

    else:

        colunas_municipio = [col for col in df.columns if col.startswith("CODMUNRES_")]

        if len(colunas_municipio) == 0:
            st.warning(
                """
                Não foram encontradas informações de município na base carregada.
                Verifique se existe a coluna `CODMUNRES` ou colunas dummizadas iniciadas por `CODMUNRES_`.
                """
            )
            st.stop()

        df_municipios = df.copy()
        df_municipios["municipio_cod"] = "referencia"

        for col in colunas_municipio:
            codigo = col.replace("CODMUNRES_", "")
            df_municipios.loc[df_municipios[col] == 1, "municipio_cod"] = codigo

    # =========================
    # Agrupamento territorial
    # =========================

    ranking_municipios = df_municipios.groupby("municipio_cod").agg(
        nascimentos=("obito_infantil", "count"),
        obitos=("obito_infantil", "sum")
    ).reset_index()

    ranking_municipios["taxa_por_mil"] = (
        ranking_municipios["obitos"] / ranking_municipios["nascimentos"]
    ) * 1000

    # Evita municípios com pouquíssimos registros distorcendo o ranking
    ranking_municipios = ranking_municipios[
        ranking_municipios["nascimentos"] >= 100
    ]

    # Adiciona nome do município
    ranking_municipios["municipio_nome"] = ranking_municipios["municipio_cod"].map(municipios_para)
    ranking_municipios["municipio_nome"] = ranking_municipios["municipio_nome"].fillna(
        ranking_municipios["municipio_cod"]
    )

    ranking_municipios = ranking_municipios.sort_values(
        "taxa_por_mil",
        ascending=False
    )

    # =========================
    # Indicadores gerais territoriais
    # =========================

    total_municipios = ranking_municipios["municipio_cod"].nunique()
    maior_taxa = ranking_municipios["taxa_por_mil"].max()
    menor_taxa = ranking_municipios["taxa_por_mil"].min()

    col1, col2, col3 = st.columns(3)

    col1.metric("Municípios analisados", total_municipios)
    col2.metric("Maior taxa por mil", f"{maior_taxa:.2f}".replace(".", ","))
    col3.metric("Menor taxa por mil", f"{menor_taxa:.2f}".replace(".", ","))

    # =========================
    # Tabela ranking
    # =========================

    st.subheader("Ranking de municípios por taxa de mortalidade infantil")

    ranking_municipios_exibicao = ranking_municipios.copy()
    ranking_municipios_exibicao["taxa_por_mil"] = ranking_municipios_exibicao["taxa_por_mil"].round(2)

    ranking_municipios_exibicao = ranking_municipios_exibicao[
        [
            "municipio_nome",
            "municipio_cod",
            "nascimentos",
            "obitos",
            "taxa_por_mil"
        ]
    ]

    ranking_municipios_exibicao = ranking_municipios_exibicao.rename(columns={
        "municipio_nome": "Município",
        "municipio_cod": "Código IBGE",
        "nascimentos": "Nascidos vivos",
        "obitos": "Óbitos infantis",
        "taxa_por_mil": "Taxa por mil"
    })

    st.dataframe(
        ranking_municipios_exibicao.head(20),
        use_container_width=True
    )

    st.markdown(
        """
        A tabela apresenta os municípios com maiores taxas de mortalidade infantil na base integrada.
        A taxa por mil permite comparar municípios de tamanhos populacionais diferentes,
        relacionando a quantidade de óbitos ao total de nascidos vivos.
        """
    )

    # =========================
    # Gráfico Top 10
    # =========================

    st.subheader("Top 10 municípios por taxa de mortalidade infantil")

    top10 = ranking_municipios.head(10).copy()

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.barh(
        top10["municipio_nome"].astype(str),
        top10["taxa_por_mil"]
    )

    ax.set_xlabel("Óbitos por mil nascidos vivos", fontsize=9)
    ax.set_ylabel("Município", fontsize=9)
    ax.set_title("Top 10 municípios por taxa de mortalidade infantil", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()

    for i, valor in enumerate(top10["taxa_por_mil"]):
        ax.text(
            valor + 0.5,
            i,
            f"{valor:.2f}".replace(".", ","),
            va="center",
            fontsize=8
        )

    plt.tight_layout()

    col_grafico, col_info = st.columns([1.2, 1])

    with col_grafico:
        st.pyplot(fig, use_container_width=False)

    with col_info:
        st.markdown(
            """
            A análise territorial ajuda a identificar municípios com maiores taxas 
            na base integrada.

            Esses resultados não devem ser interpretados isoladamente, pois podem refletir
            diferenças no volume de registros, qualidade do preenchimento, acesso aos serviços
            de saúde e características regionais.
            """
        )

    # =========================
    # Consulta por município
    # =========================

    st.subheader("Consulta por município")

    municipios_disponiveis = ranking_municipios.sort_values("municipio_nome")[
        "municipio_nome"
    ].astype(str).unique()

    municipio_selecionado = st.selectbox(
        "Selecione um município",
        options=municipios_disponiveis
    )

    dados_municipio = ranking_municipios[
        ranking_municipios["municipio_nome"].astype(str) == municipio_selecionado
    ]

    if not dados_municipio.empty:
        nasc = int(dados_municipio["nascimentos"].iloc[0])
        obt = int(dados_municipio["obitos"].iloc[0])
        taxa = dados_municipio["taxa_por_mil"].iloc[0]
        codigo = dados_municipio["municipio_cod"].iloc[0]

        st.markdown(f"**Código IBGE:** `{codigo}`")

        col1, col2, col3 = st.columns(3)

        col1.metric("Nascidos vivos", formatar_numero(nasc))
        col2.metric("Óbitos infantis", formatar_numero(obt))
        col3.metric("Taxa por mil", f"{taxa:.2f}".replace(".", ","))

    st.info(
        """
        Em uma versão futura, essa análise pode ser complementada com mapas geográficos,
        permitindo visualizar a distribuição espacial da mortalidade infantil no estado do Pará.
        """
    )

# =====================================================
# PÁGINA 6 — SIMULADOR DE RISCO
# =====================================================

elif pagina == "Simulador de risco":

    st.header("6. Simulador de risco")

    st.warning(
        """
        Este simulador é uma versão acadêmica para demonstração. O resultado não representa
        diagnóstico clínico nem deve ser utilizado para decisão individual em saúde.
        """
    )

    st.markdown(
        """
        O simulador utiliza o modelo final **XGBoost V2** para estimar a probabilidade de um registro
        estar associado ao óbito infantil, com base nas características informadas.

        As variáveis escolhidas são transformadas para o mesmo formato utilizado no treinamento do modelo.
        """
    )

    # =========================
    # Cenários prontos para demonstração
    # =========================

    st.subheader("Cenários prontos para demonstração")

    caminho_exemplos = "outputs/exemplos_demonstracao.csv"

    if os.path.exists(caminho_exemplos):

        df_exemplos = pd.read_csv(caminho_exemplos)

        # Garantir que a probabilidade esteja como número
        if "probabilidade_%" in df_exemplos.columns:
            df_exemplos["probabilidade_%"] = pd.to_numeric(
                df_exemplos["probabilidade_%"],
                errors="coerce"
            )

        st.markdown(
            """
            Os cenários abaixo foram gerados a partir de combinações testadas previamente no modelo.
            Eles ajudam a demonstrar como diferentes perfis podem resultar em probabilidades estimadas
            distintas no simulador.
            """
        )

        abas = st.tabs(["Baixa", "Moderada", "Elevada"])

        for aba, faixa in zip(abas, ["Baixa", "Moderada", "Elevada"]):

            with aba:
                df_faixa = df_exemplos[df_exemplos["faixa"] == faixa].copy()

                if df_faixa.empty:
                    st.warning(f"Nenhum cenário encontrado para a faixa {faixa}.")
                else:
                    exemplo = df_faixa.iloc[0]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Faixa",
                        faixa
                    )

                    col2.metric(
                        "Probabilidade estimada",
                        f'{exemplo["probabilidade_%"]:.2f}%'.replace(".", ",")
                    )

                    col3.metric(
                        "Município",
                        exemplo["municipio_nome"]
                    )

                    st.markdown("**Características do cenário:**")

                    dados_exemplo = pd.DataFrame({
                        "Variável": [
                            "Ano de nascimento",
                            "Mês de nascimento",
                            "Município",
                            "Sexo",
                            "Peso ao nascer",
                            "Idade materna",
                            "Escolaridade da mãe",
                            "Filhos vivos anteriores",
                            "Filhos mortos anteriores",
                            "Duração da gestação",
                            "Tipo de gravidez",
                            "Tipo de parto"
                        ],
                        "Valor": [
                            exemplo["ano_nasc"],
                            exemplo["mes_nasc"],
                            exemplo["municipio_nome"],
                            exemplo["sexo"],
                            exemplo["peso"],
                            exemplo["idade_mae"],
                            exemplo["escolaridade_mae"],
                            exemplo["filhos_vivos"],
                            exemplo["filhos_mortos"],
                            exemplo["gestacao"],
                            exemplo["gravidez"],
                            exemplo["parto"]
                        ]
                    })

                    st.dataframe(
                        dados_exemplo,
                        use_container_width=True,
                        hide_index=True
                    )

        st.info(
            """
            Esses cenários são úteis para demonstração, mas não representam casos reais individuais.
            A classificação baixa, moderada ou elevada é apenas interpretativa e foi definida para
            facilitar a leitura do comportamento do modelo.
            """
        )

    else:

        st.warning(
            """
            O arquivo `outputs/exemplos_demonstracao.csv` ainda não foi encontrado.
            Para gerar cenários prontos, execute o script `testar_cenarios.py`.
            """
        )

    st.divider()

    st.subheader("Informe as características do registro")

    # =========================
    # Lista de municípios
    # =========================

    municipios_opcoes = {
        nome: codigo
        for codigo, nome in municipios_para.items()
        if codigo not in ["referencia"]
    }

    municipios_opcoes = dict(sorted(municipios_opcoes.items()))

    col1, col2 = st.columns(2)

    with col1:
        ano_nasc = st.selectbox(
            "Ano de nascimento",
            options=list(range(2013, 2025)),
            index=list(range(2013, 2025)).index(2023)
        )

        mes_nasc = st.selectbox(
            "Mês de nascimento",
            options=list(range(1, 13)),
            index=0
        )

        municipio_nome = st.selectbox(
            "Município de residência",
            options=list(municipios_opcoes.keys()),
            index=list(municipios_opcoes.keys()).index("Belém") if "Belém" in municipios_opcoes else 0
        )

        municipio = municipios_opcoes[municipio_nome]

        sexo = st.selectbox(
            "Sexo do recém-nascido",
            options=["feminino", "masculino", "ignorado"],
            index=1
        )

        peso = st.selectbox(
            "Peso ao nascer",
            options=["baixo", "insuficiente", "adequado", "alto", "ignorado"],
            index=2
        )

        idade_mae = st.selectbox(
            "Idade materna",
            options=["menor_20", "20-34", "35-49", "50+", "ignorado"],
            index=1
        )

    with col2:
        escolaridade_mae = st.selectbox(
            "Escolaridade da mãe",
            options=["alta", "media", "baixa", "ignorado"],
            index=1
        )

        filhos_vivos = st.selectbox(
            "Filhos vivos anteriores",
            options=["0", "1-3", "4-6", "7+", "ignorado"],
            index=1
        )

        filhos_mortos = st.selectbox(
            "Filhos mortos anteriores",
            options=["0", "1-3", "4-6", "7+", "ignorado"],
            index=0
        )

        gestacao = st.selectbox(
            "Duração da gestação",
            options=["pre-termo", "a-termo", "pos-termo", "ignorado"],
            index=1
        )

        gravidez = st.selectbox(
            "Tipo de gravidez",
            options=["unica", "multipla", "ignorado"],
            index=0
        )

        parto = st.selectbox(
            "Tipo de parto",
            options=["cesareo", "normal", "ignorado"],
            index=1
        )

    # =========================
    # Botão de predição
    # =========================

    if st.button("Estimar probabilidade"):

        entrada = criar_entrada_modelo(
            ano_nasc=ano_nasc,
            mes_nasc=mes_nasc,
            municipio=municipio,
            sexo=sexo,
            peso=peso,
            idade_mae=idade_mae,
            escolaridade_mae=escolaridade_mae,
            filhos_vivos=filhos_vivos,
            filhos_mortos=filhos_mortos,
            gestacao=gestacao,
            gravidez=gravidez,
            parto=parto
        )

        try:
            probabilidade = modelo.predict_proba(entrada)[0][1]
            percentual = probabilidade * 100
            classificacao = classificar_probabilidade(percentual)

            st.subheader("Resultado da simulação")

            col_a, col_b, col_c = st.columns(3)

            col_a.metric(
                "Probabilidade estimada",
                f"{percentual:.2f}%".replace(".", ",")
            )

            col_b.metric(
                "Classificação interpretativa",
                classificacao
            )

            col_c.metric(
                "Modelo utilizado",
                "XGBoost V2"
            )

            st.caption(
                """
                Critério interpretativo utilizado: baixa probabilidade estimada quando abaixo de 5%;
                moderada entre 5% e 15%; elevada acima de 15%. Essas faixas são demonstrativas
                e não representam critérios clínicos oficiais.
                """
            )

            st.markdown(
                f"""
                **Município selecionado:** {municipio_nome}  
                **Código IBGE:** `{municipio}`
                """
            )

            if percentual < 5:
                st.success(
                    """
                    A simulação apresentou baixa probabilidade estimada de associação ao óbito infantil.
                    Ainda assim, o resultado deve ser interpretado apenas como demonstração acadêmica.
                    """
                )
            elif percentual < 15:
                st.warning(
                    """
                    A simulação apresentou probabilidade estimada moderada. Esse resultado indica que,
                    para o perfil informado, o modelo identificou alguns fatores associados ao evento.
                    """
                )
            else:
                st.error(
                    """
                    A simulação apresentou probabilidade estimada elevada. Esse resultado deve ser
                    interpretado com cautela e não representa diagnóstico ou decisão clínica.
                    """
                )

            st.info(
                """
                Essa classificação é apenas interpretativa e serve para demonstração acadêmica.
                O resultado deve ser analisado considerando as limitações do modelo, o desbalanceamento
                da base e o contexto epidemiológico.
                """
            )

            with st.expander("Visualizar entrada enviada ao modelo"):
                st.dataframe(entrada, use_container_width=True)

        except Exception as erro:
            st.error("Ocorreu um erro ao realizar a predição.")
            st.write(erro)

    # =========================
    # Observação técnica
    # =========================

    st.markdown(
        """
        **Observação técnica:** algumas categorias não aparecem explicitamente nas colunas do modelo
        porque ficaram como categorias de referência após a codificação das variáveis. Nesses casos,
        manter as colunas zeradas representa corretamente essa categoria de referência.
        """
    )

    st.markdown(
        """
        **Uso responsável:** o simulador não deve ser utilizado para avaliar casos individuais reais.
        Sua finalidade é demonstrar o funcionamento do modelo final dentro de um sistema interativo.
        """
    )

# =====================================================
# PÁGINA 8 — ARQUITETURA E MLOPS
# =====================================================

elif pagina == "Arquitetura e MLOps":

    st.header("7. Arquitetura de dados e MLOps básico")

    st.markdown(
        """
        Esta seção apresenta a organização técnica do projeto, desde a origem dos dados até
        o deploy do modelo em uma aplicação interativa. O objetivo é demonstrar que a solução
        foi estruturada de forma reprodutível, organizada e alinhada a práticas básicas de MLOps.
        """
    )

    # =========================
    # Arquitetura geral
    # =========================

    st.subheader("Arquitetura geral da solução")

    st.markdown(
        """
        O projeto seguiu um fluxo de ciência de dados aplicado à saúde pública, partindo dos
        dados brutos dos sistemas SIM e SINASC até a disponibilização dos resultados em um
        dashboard interativo.

        **Fluxo geral da solução:**

        `DATASUS → SIM/SINASC → Tratamento e padronização → Integração das bases → Base analítica → Modelo XGBoost → Dashboard Streamlit`
        """
    )

    arquitetura = pd.DataFrame({
        "Camada": [
            "Origem dos dados",
            "Data Lake",
            "Processamento",
            "Base analítica",
            "Modelagem",
            "Deploy",
            "Consumo"
        ],
        "Descrição no projeto": [
            "Dados públicos dos sistemas SIM e SINASC, obtidos a partir do DATASUS.",
            "Representa a camada de dados brutos, antes do tratamento e padronização.",
            "Etapas de limpeza, conversão de datas, seleção de variáveis e padronização.",
            "Base integrada utilizada para análises, modelagem e visualizações.",
            "Treinamento e avaliação do modelo final XGBoost V2.",
            "Aplicação interativa desenvolvida em Streamlit.",
            "Visualização dos indicadores, fatores associados e simulações de risco."
        ]
    })

    st.dataframe(arquitetura, use_container_width=True, hide_index=True)

    # =========================
    # Data Lake, Data Warehouse, ETL e ELT
    # =========================

    st.subheader("Conceitos de arquitetura de dados aplicados")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Data Lake**

            No contexto do projeto, o Data Lake corresponde à camada onde os dados brutos
            do SIM e do SINASC são armazenados antes do tratamento. Essa camada preserva
            os dados originais para consulta, auditoria e reprocessamento.
            """
        )

        st.markdown(
            """
            **ETL**

            O projeto utiliza uma lógica de ETL: os dados são extraídos dos sistemas,
            transformados por meio de limpeza, padronização e integração, e posteriormente
            carregados em uma base analítica.
            """
        )

    with col2:
        st.markdown(
            """
            **Data Warehouse**

            A base integrada final funciona como uma camada analítica, semelhante a um
            Data Warehouse, organizada para consultas, indicadores, modelagem e visualização.
            """
        )

        st.markdown(
            """
            **ELT**

            Em uma arquitetura futura, o projeto poderia utilizar ELT, carregando os dados
            brutos primeiro em uma plataforma centralizada e realizando as transformações
            posteriormente dentro do ambiente analítico.
            """
        )

    # =========================
    # Pipeline
    # =========================

    st.subheader("Pipeline do projeto")

    pipeline = pd.DataFrame({
        "Etapa": [
            "1. Coleta",
            "2. Tratamento",
            "3. Integração",
            "4. Feature Engineering",
            "5. Modelagem",
            "6. Avaliação",
            "7. Interpretabilidade",
            "8. Deploy"
        ],
        "Descrição": [
            "Obtenção dos dados públicos do SIM e SINASC.",
            "Padronização das variáveis, datas e registros.",
            "Pareamento determinístico entre nascimentos e óbitos.",
            "Criação da variável-alvo `obito_infantil` e codificação das variáveis categóricas.",
            "Treinamento do modelo XGBoost V2.",
            "Uso de métricas como AUC-ROC, precision, recall, F1-score e Average Precision.",
            "Aplicação de SHAP para interpretação dos fatores associados.",
            "Disponibilização do modelo e resultados em dashboard Streamlit."
        ]
    })

    st.dataframe(pipeline, use_container_width=True, hide_index=True)

    # =========================
    # MLOps básico
    # =========================

    st.subheader("MLOps básico")

    st.markdown(
        """
        O projeto adota práticas básicas de MLOps para permitir organização, reprodutibilidade
        e continuidade do desenvolvimento. Embora não se trate de uma infraestrutura de produção
        completa, a solução foi organizada para facilitar execução, manutenção e apresentação.
        """
    )

    mlops = pd.DataFrame({
        "Elemento": [
            "Versionamento de código",
            "Organização de pastas",
            "Modelo salvo",
            "Colunas do modelo",
            "Dependências",
            "Execução do sistema",
            "Reprodutibilidade"
        ],
        "Aplicação no projeto": [
            "Código organizado para publicação no GitHub.",
            "Separação entre `data/`, `models/`, `outputs/` e arquivo `app.py`.",
            "Modelo final salvo como `modelo_xgboost_v2.json`.",
            "Colunas utilizadas no treinamento salvas em `colunas_modelo.pkl`.",
            "Dependências listadas em `requirements.txt`.",
            "Aplicação executada com `streamlit run app.py`.",
            "Fluxo documentado para carregar base, modelo e reproduzir o dashboard."
        ]
    })

    st.dataframe(mlops, use_container_width=True, hide_index=True)

    # =========================
    # Estrutura do projeto
    # =========================

    st.subheader("Estrutura de arquivos")

    st.code(
        """
Projeto_NAP2/
│
├── app.py
├── requirements.txt
│
├── data/
│   └── base_modelagem.parquet
│
├── models/
│   ├── modelo_xgboost_v2.json
│   └── colunas_modelo.pkl
│
├── outputs/
│   ├── shap_summary.png
│   ├── cenarios_testados.csv
│   └── exemplos_demonstracao.csv
│
└── notebooks/
    └── modelagem_xgboost.ipynb
        """,
        language="text"
    )

    # =========================
    # Fluxo de treinamento e validação
    # =========================

    st.subheader("Fluxo de treinamento e validação")

    st.markdown(
        """
        O fluxo de treinamento foi realizado no notebook de modelagem, com separação entre
        dados de treino e teste. Após a avaliação do desempenho, o modelo final foi salvo
        em formato nativo do XGBoost e integrado ao dashboard.

        **Resumo do fluxo:**

        1. Carregamento da base integrada;
        2. Separação entre variáveis preditoras e variável-alvo;
        3. Divisão em treino e teste;
        4. Treinamento do XGBoost V2;
        5. Avaliação com métricas avançadas;
        6. Interpretação com SHAP;
        7. Salvamento do modelo final;
        8. Carregamento do modelo no Streamlit.
        """
    )

    st.info(
        """
        Em uma evolução futura, o pipeline poderia ser automatizado com ferramentas como MLflow,
        DVC, GitHub Actions ou orquestradores de workflow, permitindo rastreamento de experimentos,
        versionamento de dados e retreinamento periódico.
        """
    )

    st.success(
        """
        Esta página cobre os requisitos de arquitetura de dados, pipeline reprodutível,
        MLOps básico, organização do código e integração do modelo ao sistema.
        """
    )
# =====================================================
# PÁGINA 9 — ÉTICA, LIMITAÇÕES E MONITORAMENTO
# =====================================================

elif pagina == "Ética e limitações":

    st.header("8. Ética, limitações e monitoramento")

    st.markdown(
        """
        Esta seção apresenta os principais cuidados éticos, limitações metodológicas e necessidades
        de monitoramento associadas ao uso de modelos de aprendizado de máquina em dados de saúde pública.
        """
    )

    # =========================
    # LGPD e dados públicos
    # =========================

    st.subheader("Uso de dados públicos e LGPD")

    st.markdown(
        """
        O projeto utiliza dados públicos disponibilizados por sistemas oficiais de informação em saúde,
        especialmente SIM e SINASC. A análise foi conduzida com dados agregados e sem identificação
        direta dos indivíduos.

        Mesmo utilizando dados públicos, é necessário respeitar princípios da Lei Geral de Proteção
        de Dados Pessoais, como finalidade, necessidade, transparência e segurança no tratamento das
        informações.
        """
    )

    st.info(
        """
        O dashboard não deve expor informações individualizadas, nomes, documentos ou qualquer elemento
        que permita identificar pessoas específicas.
        """
    )

    # =========================
    # Uso responsável do modelo
    # =========================

    st.subheader("Uso responsável do modelo")

    st.markdown(
        """
        O modelo XGBoost V2 foi desenvolvido com finalidade acadêmica e exploratória. Ele estima a
        probabilidade de associação de um registro ao óbito infantil com base em padrões históricos
        observados na base integrada.

        Portanto, os resultados não devem ser utilizados como diagnóstico clínico, decisão individual
        em saúde ou substituição da avaliação de profissionais e gestores públicos.
        """
    )

    responsabilidades = pd.DataFrame({
        "Ponto de atenção": [
            "Diagnóstico individual",
            "Decisão clínica",
            "Gestão pública",
            "Interpretação dos resultados",
            "Comunicação dos achados"
        ],
        "Orientação": [
            "O modelo não deve ser usado para diagnosticar casos individuais.",
            "O resultado não substitui avaliação profissional ou protocolos oficiais.",
            "Os achados podem apoiar análises agregadas e planejamento em saúde pública.",
            "As probabilidades devem ser interpretadas com cautela.",
            "A comunicação deve evitar conclusões determinísticas ou culpabilização de grupos."
        ]
    })

    st.dataframe(
        responsabilidades,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # Viés algorítmico
    # =========================

    st.subheader("Viés algorítmico e desigualdades territoriais")

    st.markdown(
        """
        Modelos de aprendizado de máquina podem reproduzir desigualdades presentes nos dados históricos.
        No contexto da mortalidade infantil, diferenças territoriais, socioeconômicas, assistenciais e
        de qualidade do registro podem influenciar os resultados.

        Por isso, variáveis como município de residência, peso ao nascer, idade materna e duração da
        gestação devem ser interpretadas como fatores associados ao padrão observado, e não como causas
        isoladas ou determinísticas.
        """
    )

    st.warning(
        """
        Um risco importante é transformar desigualdades históricas em classificações automáticas sem
        considerar o contexto social, territorial e assistencial.
        """
    )

    # =========================
    # Limitações metodológicas
    # =========================

    st.subheader("Limitações metodológicas")

    limitacoes = pd.DataFrame({
        "Limitação": [
            "Dados administrativos",
            "Qualidade do preenchimento",
            "Pareamento determinístico",
            "Desbalanceamento da base",
            "Ausência de variáveis clínicas detalhadas",
            "Ano de 2014 removido",
            "Modelo não temporal"
        ],
        "Descrição": [
            "Os dados dependem da qualidade dos registros nos sistemas oficiais.",
            "Campos ausentes, ignorados ou inconsistentes podem afetar a análise.",
            "A integração SIM/SINASC pode não capturar todos os vínculos possíveis.",
            "A classe óbito infantil é minoritária, exigindo métricas além da acurácia.",
            "A base não contém todos os fatores clínicos e sociais que influenciam o fenômeno.",
            "O ano de 2014 foi tratado como outlier técnico e removido da modelagem.",
            "O XGBoost não é um modelo específico de previsão temporal."
        ]
    })

    st.dataframe(
        limitacoes,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # Monitoramento e drift
    # =========================

    st.subheader("Monitoramento e drift")

    st.markdown(
        """
        Caso o modelo fosse utilizado em um ambiente real, seria necessário monitorar continuamente
        seu desempenho e a distribuição dos dados de entrada. Mudanças no perfil dos nascimentos,
        nas características dos registros ou na qualidade do preenchimento podem causar o chamado
        **drift de dados**.
        """
    )

    monitoramento = pd.DataFrame({
        "Elemento monitorado": [
            "Distribuição das variáveis",
            "Proporção da classe óbito infantil",
            "Desempenho do modelo",
            "Dados ausentes ou ignorados",
            "Mudanças territoriais",
            "Atualização dos dados"
        ],
        "Exemplo de acompanhamento": [
            "Comparar peso, gestação, idade materna e município ao longo dos anos.",
            "Verificar se a taxa observada muda de forma relevante.",
            "Reavaliar AUC-ROC, recall, precision, F1-score e Average Precision.",
            "Acompanhar crescimento de campos ignorados ou incompletos.",
            "Observar alterações no padrão dos municípios.",
            "Retreinar o modelo com novos anos do SIM e SINASC."
        ]
    })

    st.dataframe(
        monitoramento,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        O monitoramento é fundamental para garantir que o modelo continue coerente com os dados mais
        recentes e não passe a produzir resultados pouco confiáveis após mudanças no padrão dos registros.
        """
    )

    # =========================
    # Recomendações futuras
    # =========================

    st.subheader("Recomendações futuras")

    st.markdown(
        """
        Como evolução do projeto, recomenda-se:

        - Atualizar periodicamente a base com novos anos do SIM e SINASC;
        - Realizar validação temporal do modelo;
        - Testar métodos específicos de séries temporais;
        - Avaliar estratégias adicionais para lidar com desbalanceamento;
        - Ampliar a análise de interpretabilidade;
        - Documentar versões da base, do modelo e dos experimentos;
        - Explorar ferramentas como MLflow, DVC ou GitHub Actions para MLOps.
        """
    )

    st.success(
        """
        Esta seção reforça que o modelo deve ser utilizado com responsabilidade, transparência e
        consciência das limitações dos dados e do contexto social da mortalidade infantil.
        """
    )

    st.success("Sistema funcional para demonstração acadêmica do NAP2.")