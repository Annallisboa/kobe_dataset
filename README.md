# kobe_dataset
Esse repositório contém o Projeto da disciplina de engenharia de ML do curso de MIT em Inteligência Artificial do Infnet.
A ideia do projeto é validar o conhecimento sobre Engenharia de Machine Learning, aplicando conceitos de AutoML, MLOps, Visualização de Dados e Estrutura de Projetos. 

## Prevendo arremessos do Kobe 
Nesse projetos iremos trabalhar os datasets disponíveis no  [https://www.kaggle.com/c/kobe-bryant-shot-selection/data]. A ideia é criar toda a estrutura baseada nos conceitos acima. 
As pastas estão divididas em: 
- **Code:** onde contém os códigos, execuções e requirements
- **Data:** dados brutos e processados 
- **Docs:** documentação
  
![Estrutura do projeto](https://github.com/Annallisboa/kobe_dataset/blob/main/framework%20tdsp.png)

  
## Como as ferramentas Streamlit, MLFlow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines descritos anteriormente? A resposta deve abranger os seguintes aspectos:

- **Rastreamento de Experimentos**: O MLFLOW permite o armazenamento de vários coisas automaticamente, sendo elas: parametros, métricas, artefatos e modelos treinados. Além disso, permite a comparação de métricas para decidir qual modelo é melhor. 
- **Funções de Treinamento**: o Pycaret ajuda bastante na hora de treinar vários modelos ao mesmo tempo (apesar do consumo excessivo de CPU's), basta usar um comando como setup() para treinar vários modelos automaticamente. O sckit-learn ajuda no controle da pipeline de modelagem no pré-processamneto, validação cruzada e treino.
- **Monitoramento da Saúde do Modelo**: O MLFLOW possui o histórico das métircas dos modelos, permitindo verificar como ele está performando ao longo do tempo. Para ter uma acompanhamento mais visual e disponivel para outros times, o Streamlit pode ser usado para montar dashboard e acompanhamentos das métricas totalmente customizavéis em tempo real. 
- **Atualização do Modelo:** com Pycaret podemos re-treinar os dados juntamente com o MLFLOW que possui o controle das versões anteriores, podendo ser substituido, atualizado ou até mesmo reprocessar com novos dados.
- **Deployment**: O MLFLOW consegue servir o modelo com uma API através de um unico comando, além disso, você pode "unir" ele ao Streamlit para disponibilizr o mesmo via uma interface simples da web, onde o usuário pode interagir com o modelo. 

