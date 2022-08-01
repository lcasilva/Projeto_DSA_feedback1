# Projeto Feedback 1
# Machine Learning em Logística: Prevendo o consumo de energia de carros elétricos

## Definição do Problema de Negócio

# Uma empresa de transporte e logística deseja migrar sua frota para carros elétricos
# com o objetivo de reduzir custos.
# Antes de tomar a decisão, a empresa gostaria de prever o consumo de energia de carros
# elétricos com base em diversos fatores.
# Para tanto, usaremos o seguinte dataset com dados reais disponíveis publicamente:
# https://data.mendeley.com/datasets/tb9yrptydn/2

# O dataset em pauta inclui todos os carros totalmente elétricos (apenas automóveis de 
# passageiros cuja finalidade principal é o transporte de pessoas) no mercado primário 
# que foram obtidos de materiais oficiais (especificações técnicas e catálogos) fornecidos
# por fabricantes de automóveis com licença para vender carros na Polônia. Esses materiais 
# foram baixados de seus sites oficiais. 
# O dataset é composto por 53 carros elétricos (cada variante de um modelo – que difere em 
# termos de capacidade da bateria, potência do motor etc) e 22 variáveis (25 variáveis, 
# incluindo marca, modelo e “nome do carro” mesclando estes dois anteriores).

# Créditos:
# Hadasik, Bartłomiej; Kubiczek, Jakub (2021), “Dataset of electric passenger cars with 
# their specifications”, Mendeley Data, V2, doi: 10.17632/tb9yrptydn.2


## Objetivo 
# Construir um modelo de Machine Learning capaz de prever o consumo de energia de carros 
# elétricos com base em diversos fatores, tais como, o tipo e número de motores elétricos 
# do veículo, o peso do veículo, a capacidade de carga, entre outros atributos.


## Etapas do Processo de Construção de ML: 
# 1. Definição do Problema de Negócio (descrição supracitada);
# 2. Coleta dos dados;
# 3. Análise Exploratória dos dados;
# 4. Pré-Processamento dos dados (Limpeza, Organização dos dados e Seleção de Variáveis);
# 5. Preperação dos dados de treino e dados de teste;
# 6. Construção, treinamento, avaliação e otimização do Modelo de ML.


setwd("C:/Users/User/Cursos/DSA/FCD/1-BigDataRAzure/Cap20-Projetos_com_Feedback/Projeto_Feedback1")
getwd()

# Instalando pacote para leitura de arquivo excell
# install.packages("readxl")

# Carregando pacotes necessários
library(readxl) # Usado para leitura de arquivos excell
library(randomForest) # Usado para Seleção de Variáveis
library(caret) # Usado para otimização do modelo de machine learning

# 2. Coleta dos dados
dados <- read_excel("FEV-data-Excel.xlsx")
View(dados)
dim(dados)
str(dados)
summary(dados)


## 3. Análise Exploratória dos dados

# Renomeando as variáveis do dataset
names(dados) <- c("NomeCarro","Fabricante","Modelo","PrecoMinimoBruto","PotenciaMotor","TorqueMaximo",
                  "TipoFreios","SistemaTransmissao","CapacidadeBateria","AlcanceWLTP",
                  "DistanciaEntreEixos","Comprimento","Largura","Altura","PesoVazioMinimo",
                  "PesoBrutoPermitido","CapacidadeMaximaCarga","NumeroAssentos","NumeroPortas",
                  "TamanhoPneus","VelocidadeMaxima","CapacidadePortaMalas","Aceleracao_0_a_100kph",
                  "PotenciaMaximaCarregamentoDC","ConsumoEnergia")
names(dados)
View(dados)

# Analisando os dados podemos desde já reconhecer as variáveis target e preditoras do modelo:
# ConsumoEnergia = Variável target (variável dependente, o que se deseja prever para novos dados)
# NomeCarro, PrecoMinimo, PotenciaMotor, TorqueMaximo etc =  variáveis preditoras (variáveis independentes) 

# Verificando se os dados possuem valores missing (valores ausentes)
any(is.na(dados))

# Obtendo o número de linhas que possuem colunas não preenchidas (variáveis com valores nulos)
# Uma linha = um case (um caso) = um registro = uma oberservação 
sum(!complete.cases(dados))

# Agora, obtendo o número de linhas com colunas válidas (todas as variáveis preenchidas)
sum(complete.cases(dados))

# Qual o percentual de casos incompletos?
sum(!complete.cases(dados)) / sum(complete.cases(dados)) * 100 
# 26,19% do dataset tem valores missing (valores ausentes)

# Obtendo as variáveis com valores missing
sapply(dados, function(x) sum(is.na(x))) # ou colSums(is.na(dados))


## 4. Pré-Processamento dos dados (Limpeza, Organização dos dados e Seleção de Variáveis)
# Estratégia usada para remover linhas com valores missing:
# A estratégia usada pata tratamento de valores NA foi a remoção de todas as linhas que 
# contenham valores NA. Dessa forma, os dados permaneceriam mais fidedignos com a realidade. 
# Se a estratégia fosse a imputação de valores via média, mediana etc, estatisticamente 
# estaria correto, porém afetaria a integridade dos dados. Para o problema de negócio em pauta, 
# a imputação de dados, levaria a resultados fictícios. 

# Eliminando observações com valores missing
# Criando outra variável para não perder os dados originais
dadosPos <- na.omit(dados)   # ou dadosPos <- (dados[complete.cases(dados),])
sapply(dadosPos, function(x) sum(is.na(x)))
dim(dadosPos)
View(dadosPos)


## Plotando Boxplots e Histogramas de todas as variáveis numéricas
df <- as.data.frame(dadosPos)
numeric.var <- sapply(df, is.numeric) # Selecionar apenas variáveis numéricas

# Plotando histograma e boxplot de todas as variáveis numéricas do dataset
# Gráficos para análise de apenas uma variável - Boxplot e histograma
graphs <- lapply(names(df[,numeric.var]), function(x) {hist(df[,x], 100,
                                                               col="lightblue",
                                                               main=paste0("Histograma de ", x),
                                                               xlab=x);boxplot(df[,x], 
                                                                               main=paste0("Boxplot de ",x), 
                                                                               horizontal = TRUE)})
remove(df)
remove(graphs)
remove(numeric.var)


## Seleção de Variáveis mais relevantes (Feature Selection)

# Que variáveis (features) presentes em nosso conjunto de dados, devem ser usadas na criação do modelo?
# Avalidando a importância de todas as variaveis usando modelo randomForest
modelo <- randomForest(ConsumoEnergia ~ . , 
                       data = dadosPos, 
                       ntree = 100, 
                       nodesize = 10,
                       importance = TRUE) # importance = TRUE nos informa as variáveis mais importantes ...

# Plotando as variáveis por grau de importância
varImpPlot(modelo)
remove(modelo)

# Removendo as variáveis que menos contribuem ...
modelo2 <- randomForest(ConsumoEnergia ~ .
                        - CapacidadeMaximaCarga
                        - Aceleracao_0_a_100kph
                        - NomeCarro
                        - AlcanceWLTP
                        - Altura
                        - TamanhoPneus
                        - Fabricante
                        - CapacidadeBateria
                        - Modelo
                        - PotenciaMaximaCarregamentoDC
                        - NumeroAssentos
                        - TipoFreios
                        - NumeroPortas
                        , 
                        data = dadosPos, 
                        ntree = 100, 
                        nodesize = 10,
                        importance = TRUE)  

varImpPlot(modelo2)
remove(modelo2)

# Atualizando o dataset apenas com as variáveis mais relevantes
dadosPos <- dadosPos[,c("PesoBrutoPermitido",
                        "DistanciaEntreEixos",
                        "Comprimento",
                        "SistemaTransmissao",
                        "PesoVazioMinimo",
                        "Largura",
                        "PrecoMinimoBruto",
                        "TorqueMaximo",
                        "VelocidadeMaxima",
                        "PotenciaMotor",
                        "ConsumoEnergia")]
View(dadosPos)
str(dadosPos)
dim(dadosPos)


# Salvando dataset em arquivo csv
#write.csv(x = dadosPos, file = "dataset.csv", row.names = FALSE)

# Extraindo as variáveis numéricas
numeric_variable_list <- sapply(dadosPos, is.numeric)
numerical_data <- dadosPos[ ,numeric_variable_list]

# Análise Exploratória dos dados - Variáveis numéricas
# Matriz de Correlação - Explorando relacionamento entre variáveis numéricas
cor(numerical_data)

# Gerando o gráfico de correlação
#pairs(numerical_data)
pairs(numerical_data[1:5], labels = colnames(numerical_data)[1:5])
pairs(numerical_data[6:10], labels = colnames(numerical_data)[6:10])
remove(numeric_variable_list)
remove(numerical_data)


## 5. Preperação dos dados de treino e dados de teste;
# Criando a variável Index que será usada posteriormente para a divisão de dados de treino e dados de teste
dadosPos[,"index"] <- ifelse(runif(nrow(dadosPos)) < 0.7, 1, 0) # runif(n, min = 0, max = 1)
table(dadosPos$index)
round(prop.table(table(dadosPos$index)) * 100, digits = 2)

# Dados de treino e dados de teste
trainset <- dadosPos[dadosPos$index == 1, ]
testset <- dadosPos[dadosPos$index == 0, ]
nrow(trainset)
nrow(testset)

# Remoção do index dos conjuntos de dados de treino, teste e dadosPos
trainColNum <- grep('index', names(trainset)) # obtendo o índice da coluna da variável 'index'
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]
dadosPos <- dadosPos[,-trainColNum]
View(trainset)
View(testset)
remove(trainColNum)


# 6. Construção, treinamento, avaliação e otimização do Modelo de ML

# Construção e treinamento do modelo usando Algoritmo de Regressão Linear
modelo_v1 <- lm(ConsumoEnergia ~., data = trainset) 

# Avaliando o modelo_v1
summary(modelo_v1) # R-squared:  0.9078, p-value: 2.883e-06

# Construção e treinamento do modelo usando Alghoritmo de Random Forest do pacote caret
modelo_v2 <- train(ConsumoEnergia ~ ., data = trainset, method = 'rf')

# Avaliando o modelo_v2
modelo_v2 # R-squared:  0.8083237

# Ajustando o modelo_v1 (permite otimizar o modelo)
controle1 <- trainControl(method = "cv", number = 10) 
# método = 'cv' (cross validation com 10 combinações de dados de treino e teste)

modelo_v3 <- train(ConsumoEnergia ~., 
                   data = trainset,
                   method = 'lm',
                   trControl = controle1,
                   metric = "Rsquared") # Coeficiente de determinação

# Avaliando o modelo_v2
summary(modelo_v3) # R-squared:  0.9078

# Resumo dos modelos v1, v2 e v3
summary(modelo_v1) # R-squared:  0.9078
modelo_v2 # R-squared:  0.8083237
summary(modelo_v3) # R-squared:  0.9078

# Resultado
# O modelo de Regressão Linear apresenta a melhor acurácia, e a versão otimizada (versão 3)
# não apresentou aumento da acurácia. Portanto usaremos a versão v1.

# Gerando as previsões ....
previsao <- predict(modelo_v1, testset)
previsao
data.frame(testset$ConsumoEnergia, previsao, previsao - testset$ConsumoEnergia)
plot(testset$ConsumoEnergia, previsao)
