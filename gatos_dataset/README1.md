## Características principais:

1. **Preparação de Dados**:
   - Usa `ImageDataGenerator` para carregar imagens das pastas
   - Aplica augmentação de dados (rotação, zoom, flip) para melhorar a generalização
   - Divide automaticamente os dados de treino em treino (80%) e validação (20%)

2. **Modelo EfficientNetB0**:
   - Usa transfer learning com pesos pré-treinados do ImageNet
   - Adiciona camadas customizadas para classificação das 4 raças
   - Implementa dropout para evitar overfitting

3. **Treinamento em Duas Fases**:
   - **Fase 1**: Treina apenas as camadas superiores (feature extraction)
   - **Fase 2**: Fine-tuning das últimas 20 camadas do EfficientNet

4. **Callbacks Importantes**:
   - `ModelCheckpoint`: Salva o melhor modelo
   - `EarlyStopping`: Para o treino se não houver melhoria
   - `ReduceLROnPlateau`: Reduz a taxa de aprendizado quando necessário

5. **Avaliação e Visualização**:
   - Gera gráficos de acurácia e loss
   - Cria matriz de confusão
   - Mostra relatório detalhado de classificação
   - Função para fazer predições em novas imagens

## Como usar:

1. **Instalar dependências**:
```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy
```

2. **Organizar as pastas**:
```
├── treino/
│   ├── maine_coon/
│   ├── persa/
│   ├── siames/
│   └── sphynx/
└── teste/
    ├── maine_coon/
    ├── persa/
    ├── siames/
    └── sphynx/
```

3. **Executar o código**:
```bash
python efficientnet.py
```

4. **Para fazer predições em novas imagens**:
```python
# Carregar o modelo salvo
modelo = tf.keras.models.load_model('classificador_racas_gatos_final.h5')

# Fazer predição
predict_cat_breed('caminho/para/imagem_gato.jpg', modelo)
```