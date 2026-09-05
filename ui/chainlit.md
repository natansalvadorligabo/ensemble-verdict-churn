# Churn Intelligence

### Ensemble de modelos para análise de churn

Projeto acadêmico desenvolvido no contexto do **MBA em Inteligência Artificial & Big Data do ICMC-USP**.

---

## Sobre o projeto

O Churn Intelligence combina cinco modelos de aprendizado de máquina para estimar o risco de churn de clientes de e-commerce:

- K-Nearest Neighbors
- Support Vector Machine com kernel RBF
- Random Forest
- XGBoost
- Naive Bayes

Quando os modelos apresentam uma divisão de votos de 3–2, um modelo de linguagem local analisa o perfil, os votos e as confianças para arbitrar a decisão final.

## Como utilizar

Descreva o cliente naturalmente, em português ou em outro idioma. O assistente irá:

1. Extrair os atributos necessários.
2. Solicitar somente as informações ausentes.
3. Exibir o perfil para confirmação.
4. Executar os cinco classificadores.
5. Apresentar a decisão e permitir perguntas sobre o resultado.

Use `/form` para abrir o formulário manual ou `/new` para iniciar a análise de outro cliente.

## Interpretação dos rótulos

| Rótulo | Significado |
| --- | --- |
| **0** | Permanência prevista |
| **1** | Churn previsto |

As confianças apresentadas pertencem individualmente a cada modelo. A decisão representa uma estimativa estatística e não uma relação causal.

## Privacidade e execução

Todos os modelos, inclusive o árbitro Qwen, são executados localmente nos containers. Nenhuma chave do Kaggle é necessária para utilizar a demonstração.

## Responsabilidade

Este sistema é uma demonstração acadêmica e não deve ser utilizado isoladamente para decisões comerciais ou ações que afetem clientes. O dataset histórico e os modelos podem reproduzir limitações e vieses dos dados de treinamento.

---

**Autor:** Natan Salvador Ligabô  
**Programa:** MBA em Inteligência Artificial & Big Data — ICMC-USP  
**Projeto:** Churn Intelligence
