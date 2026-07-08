# Projeto NAP2 — Mortalidade Infantil no Pará

Este projeto foi desenvolvido como parte da entrega final da NAP2 da disciplina de Ciência de Dados II. O objetivo é analisar a mortalidade infantil no estado do Pará a partir da integração das bases SIM e SINASC, utilizando técnicas de aprendizado de máquina, interpretabilidade, visualização de dados e deploy com Streamlit.

## Tema do projeto

**Análise da mortalidade infantil no estado do Pará por meio da integração dos sistemas SIM e SINASC e técnicas de aprendizado de máquina.**

## Objetivo

O projeto tem como objetivo integrar dados públicos de nascimento e óbito infantil, construir uma base analítica, aplicar modelagem preditiva com XGBoost, interpretar os fatores associados com SHAP e disponibilizar os resultados em um dashboard interativo.


A proposta busca apoiar a análise da mortalidade infantil no Pará por meio de indicadores, visualizações, avaliação preditiva e storytelling orientado à tomada de decisão em saúde pública.

## Integrantes

- Eduardo Gabriel Reis Farias
- Bredson Jonh Cordeiro do Nascimento

Universidade Federal Rural da Amazônia — UFRA  
Disciplina: Ciência de Dados II

## Bases de dados utilizadas

O projeto utiliza dados públicos dos sistemas de informação em saúde:

- **SINASC** — Sistema de Informações sobre Nascidos Vivos;
- **SIM** — Sistema de Informações sobre Mortalidade.

As bases foram obtidas a partir do DATASUS e tratadas para construção de uma base integrada com foco na análise da mortalidade infantil no estado do Pará.

## Pipeline do projeto

O projeto segue um pipeline completo de ciência de dados:

1. Definição do problema;
2. Coleta dos dados;
3. Tratamento e padronização das bases SIM e SINASC;
4. Integração das bases;
5. Criação da variável-alvo `obito_infantil`;
6. Engenharia de atributos;
7. Modelagem preditiva;
8. Avaliação do modelo;
9. Interpretabilidade com SHAP;
10. Deploy em Streamlit;
11. Monitoramento, ética e análise crítica.

## Notebook do projeto

As etapas de tratamento, integração, análise exploratória e modelagem foram desenvolvidas em notebook no Google Colab.

Acesse o notebook principal:

[Notebook no Google Colab](https://colab.research.google.com/drive/1aVUTYXWfDEtdnJsnfY8AD8Zd5sPQDily?usp=sharing)

O notebook contém as seguintes etapas:

- Carregamento das bases SIM e SINASC;
- Tratamento e padronização das variáveis;
- Integração das bases;
- Criação da variável-alvo `obito_infantil`;
- Análise exploratória dos dados;
- Treinamento e avaliação do modelo XGBoost;
- Geração dos arquivos utilizados no Streamlit, como modelo salvo, colunas do modelo e gráfico SHAP.

## Modelo final

O modelo final adotado foi o **XGBoost V2**, utilizado para classificação binária da variável-alvo `obito_infantil`.

O modelo foi escolhido por sua capacidade de trabalhar com dados estruturados, lidar com relações não lineares e apresentar bom desempenho em problemas de classificação com bases desbalanceadas.

Principais resultados do modelo:

- AUC-ROC: 0,80;
- Acurácia: 0,841;
- Precision da classe óbito infantil: 0,054;
- Recall da classe óbito infantil: 0,642;
- F1-score da classe óbito infantil: 0,099;
- Average Precision: 0,101.

## Interpretabilidade

Para interpretação do modelo, foi utilizada a técnica **SHAP (SHapley Additive exPlanations)**.

A análise SHAP permitiu identificar os principais fatores associados às predições do modelo, entre eles:

- Peso ao nascer;
- Duração da gestação;
- Sexo do recém-nascido;
- Idade materna;
- Município de residência.

Esses fatores foram interpretados de forma crítica, considerando que a influência no modelo não representa, isoladamente, relação causal.

## Dashboard em Streamlit

O dashboard desenvolvido em Streamlit apresenta as seguintes seções:

- Visão geral da base;
- Modelo preditivo e avaliação;
- Fatores associados e interpretabilidade com SHAP;
- Causas de óbito infantil;
- Análise territorial por município;
- Simulador de risco;
- Projeção demonstrativa 2025/2026;
- Arquitetura de dados e MLOps;
- Ética, limitações e monitoramento.

## Estrutura do projeto

```text
projeto-nap2-mortalidade-infantil-para/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
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
└── scripts/
    └── testar_cenarios.py

#Pré-requisitos

Para executar o projeto em outra máquina, é necessário ter instalado:

Python 3.12 ou versão compatível;
Git;
Navegador de internet;
VS Code ou outro editor de código, opcional.

No Windows, durante a instalação do Python, recomenda-se marcar a opção:

Add Python to PATH


## Como clonar e executar o projeto

### 1. Clonar o repositório

No terminal, execute:

```bash
git clone https://github.com/lEduardoReis/projeto-nap2-mortalidade-infantil-para.git
```

Depois entre na pasta do projeto:

```bash
cd Projeto_NAP2
```

## 2. Criar ambiente virtual

### Windows

```bash
python -m venv venv
```

Ative o ambiente virtual:

```bash
.\venv\Scripts\activate
```

### Linux ou Mac

```bash
python3 -m venv venv
```

Ative o ambiente virtual:

```bash
source venv/bin/activate
```

## 3. Instalar as dependências

Com o ambiente virtual ativado, execute:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Verificar os arquivos necessários

Antes de rodar o sistema, confira se os seguintes arquivos existem no projeto:

```text
data/base_modelagem.parquet
models/modelo_xgboost_v2.json
models/colunas_modelo.pkl
outputs/shap_summary.png
outputs/exemplos_demonstracao.csv
```

Caso a base `base_modelagem.parquet` não esteja no repositório por limitação de tamanho, ela deve ser baixada separadamente pelo link disponibilizado pelo grupo e colocada dentro da pasta:

```text
data/
```

A estrutura esperada é:

```text
data/base_modelagem.parquet
```

## 5. Rodar o dashboard

Execute o comando:

```bash
streamlit run app.py
```

O navegador abrirá automaticamente com o dashboard. Caso não abra, acesse manualmente:

```text
http://localhost:8501
```

## Como gerar os cenários de demonstração do simulador

O simulador utiliza cenários prontos de baixa, moderada e elevada probabilidade estimada. Para gerar esses cenários, execute:

```bash
python scripts/testar_cenarios.py
```

Esse script gera os arquivos:

```text
outputs/cenarios_testados.csv
outputs/exemplos_demonstracao.csv
```

O arquivo `exemplos_demonstracao.csv` é utilizado pelo dashboard para exibir cenários prontos no simulador de risco.

## Como atualizar o modelo no sistema

Caso um novo modelo seja treinado, ele deve ser salvo em formato nativo do XGBoost:

```text
models/modelo_xgboost_v2.json
```

Também é necessário manter o arquivo com as colunas utilizadas no treinamento:

```text
models/colunas_modelo.pkl
```

Esses arquivos são carregados automaticamente pelo `app.py`.

## Observação sobre o simulador de risco

O simulador tem finalidade acadêmica e demonstrativa. Ele não realiza diagnóstico clínico, não substitui avaliação profissional e não deve ser utilizado para decisão individual em saúde.

A classificação baixa, moderada ou elevada é interpretativa:

* Baixa: abaixo de 5%;
* Moderada: entre 5% e 15%;
* Elevada: acima de 15%.

Essas faixas não representam critérios clínicos oficiais. Elas foram utilizadas apenas para facilitar a leitura dos resultados no dashboard.

## Aspectos éticos

O projeto utiliza dados públicos e anonimizados, respeitando princípios da Lei Geral de Proteção de Dados Pessoais (LGPD). Ainda assim, os resultados devem ser interpretados com cautela, considerando limitações dos registros administrativos, possibilidade de viés algorítmico e desigualdades territoriais.

O modelo deve ser entendido como ferramenta complementar de apoio à análise epidemiológica e à vigilância em saúde pública, e não como instrumento de decisão individual ou diagnóstico.

## Tecnologias utilizadas

* Python;
* Pandas;
* NumPy;
* Scikit-learn;
* XGBoost;
* SHAP;
* Matplotlib;
* Streamlit;
* Joblib;
* PyArrow.

## Comandos principais

Instalar dependências:

```bash
pip install -r requirements.txt
```

Rodar o dashboard:

```bash
streamlit run app.py
```

Gerar cenários do simulador:

```bash
python scripts/testar_cenarios.py
```

## Autores

Eduardo Gabriel Reis Farias

Universidade Federal Rural da Amazônia — UFRA
