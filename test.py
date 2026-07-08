import os
import random
import joblib
import pandas as pd
from xgboost import XGBClassifier


# =====================================================
# CONFIGURAÇÕES
# =====================================================

CAMINHO_MODELO = "models/modelo_xgboost_v2.json"
CAMINHO_COLUNAS = "models/colunas_modelo.pkl"
CAMINHO_SAIDA_COMPLETA = "outputs/cenarios_testados.csv"
CAMINHO_SAIDA_EXEMPLOS = "outputs/exemplos_demonstracao.csv"

os.makedirs("outputs", exist_ok=True)


# =====================================================
# CARREGAR MODELO E COLUNAS
# =====================================================

modelo = XGBClassifier()
modelo.load_model(CAMINHO_MODELO)

colunas_modelo = joblib.load(CAMINHO_COLUNAS)

print("Modelo e colunas carregados com sucesso.")
print(f"Quantidade de colunas do modelo: {len(colunas_modelo)}")


# =====================================================
# FUNÇÕES
# =====================================================

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
    """
    Faixas interpretativas usadas no simulador.

    Atenção:
    Essas faixas são demonstrativas, não clínicas.
    """
    if probabilidade_percentual < 5:
        return "Baixa"
    elif probabilidade_percentual < 15:
        return "Moderada"
    else:
        return "Elevada"


# =====================================================
# LISTAS DE CATEGORIAS PARA TESTE
# =====================================================

anos = [2023, 2024]
meses = [1, 6, 12]

municipios = [
    "150140",  # Belém
    "150080",  # Ananindeua
    "150680",  # Santarém
    "150040",  # Alenquer
    "150210",  # Cametá
    "150240",  # Castanhal
    "150420",  # Marabá
    "150360",  # Itaituba
    "ignorado"
]

sexos = ["feminino", "masculino", "ignorado"]

pesos = ["baixo", "insuficiente", "adequado", "alto", "ignorado"]

idades_mae = ["menor_20", "20-34", "35-49", "50+", "ignorado"]

escolaridades = ["alta", "media", "baixa", "ignorado"]

filhos_vivos_lista = ["0", "1-3", "4-6", "7+", "ignorado"]

filhos_mortos_lista = ["0", "1-3", "4-6", "7+", "ignorado"]

gestacoes = ["pre-termo", "a-termo", "pos-termo", "ignorado"]

gravidezes = ["unica", "multipla", "ignorado"]

partos = ["cesareo", "normal", "ignorado"]


municipios_nomes = {
    "150140": "Belém",
    "150080": "Ananindeua",
    "150680": "Santarém",
    "150040": "Alenquer",
    "150210": "Cametá",
    "150240": "Castanhal",
    "150420": "Marabá",
    "150360": "Itaituba",
    "ignorado": "Ignorado"
}


# =====================================================
# GERAR CENÁRIOS ALEATÓRIOS
# =====================================================

quantidade_cenarios = 50000

resultados = []

print(f"Gerando e testando {quantidade_cenarios} cenários...")

for i in range(quantidade_cenarios):

    ano_nasc = random.choice(anos)
    mes_nasc = random.choice(meses)
    municipio = random.choice(municipios)
    sexo = random.choice(sexos)
    peso = random.choice(pesos)
    idade_mae = random.choice(idades_mae)
    escolaridade_mae = random.choice(escolaridades)
    filhos_vivos = random.choice(filhos_vivos_lista)
    filhos_mortos = random.choice(filhos_mortos_lista)
    gestacao = random.choice(gestacoes)
    gravidez = random.choice(gravidezes)
    parto = random.choice(partos)

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

    probabilidade = modelo.predict_proba(entrada)[0][1] * 100
    faixa = classificar_probabilidade(probabilidade)

    resultados.append({
        "ano_nasc": ano_nasc,
        "mes_nasc": mes_nasc,
        "municipio_codigo": municipio,
        "municipio_nome": municipios_nomes.get(municipio, municipio),
        "sexo": sexo,
        "peso": peso,
        "idade_mae": idade_mae,
        "escolaridade_mae": escolaridade_mae,
        "filhos_vivos": filhos_vivos,
        "filhos_mortos": filhos_mortos,
        "gestacao": gestacao,
        "gravidez": gravidez,
        "parto": parto,
        "probabilidade_%": round(probabilidade, 2),
        "faixa": faixa
    })


df_cenarios = pd.DataFrame(resultados)

df_cenarios = df_cenarios.sort_values("probabilidade_%")


# =====================================================
# PEGAR EXEMPLOS PARA DEMONSTRAÇÃO
# =====================================================

baixo = (
    df_cenarios[df_cenarios["faixa"] == "Baixa"]
    .sort_values("probabilidade_%")
    .head(3)
)

moderado = (
    df_cenarios[df_cenarios["faixa"] == "Moderada"]
    .sort_values("probabilidade_%")
    .head(3)
)

elevado = (
    df_cenarios[df_cenarios["faixa"] == "Elevada"]
    .sort_values("probabilidade_%", ascending=False)
    .head(3)
)

exemplos = pd.concat([baixo, moderado, elevado])


# =====================================================
# SALVAR RESULTADOS
# =====================================================

df_cenarios.to_csv(CAMINHO_SAIDA_COMPLETA, index=False, encoding="utf-8-sig")
exemplos.to_csv(CAMINHO_SAIDA_EXEMPLOS, index=False, encoding="utf-8-sig")


# =====================================================
# EXIBIR RESULTADO NO TERMINAL
# =====================================================

print("\nTeste concluído.")
print(f"Arquivo completo salvo em: {CAMINHO_SAIDA_COMPLETA}")
print(f"Exemplos para demonstração salvos em: {CAMINHO_SAIDA_EXEMPLOS}")

print("\nDistribuição por faixa:")
print(df_cenarios["faixa"].value_counts())

print("\nExemplos para demonstração:")
print(exemplos.to_string(index=False))