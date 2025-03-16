Justificativa Técnica: Modelo Random Forest
Optamos pela escolha do modelo de aprendizado supervisionado baseado em Random Forest, dado que temos um dataset com características que se beneficiam de um ensemble de árvores de decisão. O Random Forest tem demonstrado excelente capacidade de lidar com diferentes tipos de variáveis e capturar interações complexas através de sua abordagem de agregação de múltiplos modelos.

A escolha foi feita considerando diversos fatores técnicos:

Robustez contra overfitting através do bagging (Bootstrap Aggregating)
Capacidade natural de lidar com dados não-lineares e categóricos sem necessidade de transformações extensivas
Feature importance nativa para interpretabilidade do modelo
Paralelização eficiente do treinamento por natureza
Menor necessidade de tuning de hiperparâmetros comparado a redes neurais
Com isso, ao fim do treinamento do modelo, vamos utilizar os seguintes parâmetros para averiguação:

Out-of-bag error (OOB) para validação cruzada implícita
Feature importance scores para interpretabilidade
Accuracy, Precision e Recall para classificação binária de fraudes
Matriz de confusão para análise detalhada dos erros
Bibliotecas utilizadas
Para implementação deste modelo, utilizamos:

Pandas para leitura dos CSV e manipulação dos dados
Matplotlib e Seaborn para visualização e análise exploratória
Scikit-learn como framework principal para construção e treinamento do Random Forest
NumPy para operações numéricas eficientes
Nossa meta para este modelo é alcançar uma acurácia superior a 97% mantendo um baixo índice de falsos positivos e negativos.

Conceitos aplicados ao Random Forest
Para otimizar o modelo, implementamos várias técnicas específicas:

Bootstrap Aggregating (Bagging) para redução de variância
Feature randomization para garantir diversidade entre as árvores
Estratificação na amostragem para lidar com classes desbalanceadas
Cross-validation para validação robusta do modelo
A escolha de Random Forest sobre redes neurais se justifica principalmente pelos seguintes aspectos:

Menor necessidade de pré-processamento dos dados
Interpretabilidade natural do modelo
Robustez a outliers e dados ruidosos
Menor risco de overfitting
Treinamento mais rápido e menos intensivo computacionalmente
Além disso, utilizamos técnicas específicas de pré-processamento adequadas para Random Forests, incluindo:

Encoding mínimo de variáveis categóricas (o algoritmo lida naturalmente com elas)
Detecção automática de features importantes
Amostragem estratificada para classes desbalanceadas
As visualizações com Matplotlib e Seaborn nos permitem analisar:

Importância relativa das features
Curvas ROC e PR
Matriz de confusão
Out-of-bag error ao longo do treinamento
