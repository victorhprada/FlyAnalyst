# FlyAnalyst: Análise Preditiva de Atrasos em Voos

## 📊 Sobre o Projeto

FlyAnalyst é uma solução de análise preditiva desenvolvida para otimizar operações aeroportuárias através da previsão de atrasos em voos. Utilizando técnicas avançadas de machine learning, o sistema analisa múltiplos fatores que influenciam os atrasos, permitindo uma gestão mais eficiente de recursos e melhor experiência para os passageiros.

### 💼 Impacto no Negócio

- **Redução de Custos Operacionais**: Previsão antecipada permite melhor alocação de recursos
- **Satisfação do Cliente**: Informações mais precisas sobre possíveis atrasos
- **Eficiência Operacional**: Otimização do planejamento de voos e equipes
- **Tomada de Decisão Baseada em Dados**: Insights acionáveis para gestão aeroportuária

## 🔍 Características Analisadas

O modelo considera diversos fatores importantes:
- Companhia aérea
- Tipo de aeronave
- Voos Schengen vs. não-Schengen
- Horários de chegada e partida
- Sazonalidade (feriados, fins de semana)
- Origem do voo
- Padrões históricos de atrasos

## 📈 Performance do Modelo

- **R² Score**: 0.64 (64% de precisão na previsão de atrasos)
- **RMSE**: 13.79 minutos
- **MAE**: 11.05 minutos

### 🔝 Principais Fatores de Influência

1. Companhia aérea (52.7% de importância)
2. Status de feriado (14.8%)
3. Tipo de aeronave (10.0%)
4. Horário de chegada (3.9%)

## 🛠 Requisitos Técnicos

```python
# Principais dependências
pandas==2.2.1
scikit-learn==1.6.0
numpy==2.2.1
matplotlib==3.10.0
seaborn==0.13.2
yellowbrick==1.5
```

## 🚀 Como Usar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/FlyAnalyst.git
cd FlyAnalyst
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o script principal:
```bash
python first.py
```

4. O modelo treinado será salvo como 'model_grid_search.pkl'

## 📊 Estrutura de Dados

O sistema espera um arquivo CSV ('flights.csv') com as seguintes colunas:
- flight_id: ID único do voo
- airline: Código da companhia aérea
- aircraft_type: Tipo de aeronave
- schengen: Status Schengen do voo
- origin: Aeroporto de origem
- arrival_time: Horário de chegada
- departure_time: Horário de partida
- day: Dia do ano (1-365)
- year: Ano do voo
- is_holiday: Indicador de feriado
- delay: Atraso em minutos (variável alvo)

## 🔄 Pipeline de Modelagem

1. **Pré-processamento**:
   - Codificação de variáveis categóricas
   - Tratamento de dados temporais
   - Engenharia de features

2. **Seleção de Features**:
   - Análise de importância de features
   - Seleção das 13 características mais relevantes

3. **Otimização de Hiperparâmetros**:
   - Grid Search com validação cruzada
   - Otimização de parâmetros do Random Forest

4. **Avaliação e Persistência**:
   - Métricas de performance
   - Salvamento do modelo otimizado

## 📈 Visualizações

O sistema gera diversos gráficos para análise:
- Distribuição de atrasos por companhia aérea
- Padrões temporais de atrasos
- Importância das features
- Gráficos de erro de predição

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests ou abrir issues para melhorias.

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## 📧 Contato

Para questões comerciais ou técnicas, entre em contato através de [seu-email@dominio.com]

---
Desenvolvido com ❤️ para otimização de operações aeroportuárias