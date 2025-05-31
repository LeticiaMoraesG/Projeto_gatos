### Principais Recursos:

### 1. **Aba de Treinamento** 📊
- Configuração interativa de hiperparâmetros
- Visualização da distribuição do dataset
- Monitoramento em tempo real do treinamento
- Gráficos de acurácia e loss
- Salvamento automático do melhor modelo

### 2. **Aba de Predição** 🔮
- Upload fácil de imagens
- Visualização da imagem original
- Predição com barra de confiança
- Gráfico de probabilidades para todas as raças
- Informações sobre cada raça detectada

### 3. **Aba de Análise** 📈
- Métricas de performance do modelo
- Matriz de confusão interativa
- Análise por raça (precisão, recall, F1-score)
- Gráfico radar comparativo
- Estatísticas detalhadas

## 🚀 Como executar:

1. **Instalar dependências**:
```bash
pip install streamlit tensorflow pillow matplotlib seaborn scikit-learn pandas numpy
```

2. **Salvar o código** como `app.py`

3. **Executar o aplicativo**:
```bash
streamlit run app.py
```

## 💡 Características especiais:

- **Interface responsiva**: Adapta-se a diferentes tamanhos de tela
- **Emojis temáticos**: Cada raça tem seu emoji representativo
- **Feedback visual**: Barras de progresso e animações
- **Métricas em tempo real**: Acompanhamento do treinamento
- **Visualizações interativas**: Gráficos e tabelas dinâmicas

## 📝 Notas importantes:

1. O aplicativo salva o modelo treinado como `classificador_racas_gatos_streamlit.h5`

2. As métricas na aba "Análise" são simuladas. Em produção, você deve calcular essas métricas usando o conjunto de teste real. 

3. Detalhe: o site sópermite carregar apenas uma imagem, para carregar outra, tem que limpar o cache, fechar o site no terminal e abri-lo novamente. 