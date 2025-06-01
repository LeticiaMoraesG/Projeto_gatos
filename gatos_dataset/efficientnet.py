import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

# Configurações
TAMANHO_IMG = (224, 224)
TAMANHO_LOTE = 32
EPOCAS = 10
TAXA_APRENDIZADO = 0.001

# Caminhos das pastas
PASTA_TREINO = 'treino'
PASTA_TESTE = 'teste'

# Classes de raças
RACAS = ['maine_coon', 'persa', 'siames', 'sphynx']
NUM_RACAS = len(RACAS)

# Criar geradores de dados com augmentação para treino
gerador_treino = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2  # 20% para validação
)

# Gerador para teste (apenas normalização)
gerador_teste = ImageDataGenerator(rescale=1./255)

# Carregar dados de treino
dados_treino = gerador_treino.flow_from_directory(
    PASTA_TREINO,
    target_size=TAMANHO_IMG,
    batch_size=TAMANHO_LOTE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Carregar dados de validação
dados_validacao = gerador_treino.flow_from_directory(
    PASTA_TREINO,
    target_size=TAMANHO_IMG,
    batch_size=TAMANHO_LOTE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Carregar dados de teste
dados_teste = gerador_teste.flow_from_directory(
    PASTA_TESTE,
    target_size=TAMANHO_IMG,
    batch_size=TAMANHO_LOTE,
    class_mode='categorical',
    shuffle=False
)

# Criar modelo usando EfficientNetB0
def criar_modelo():
    # Carregar EfficientNetB0 pré-treinado (sem as camadas superiores)
    modelo_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*TAMANHO_IMG, 3)
    )
    
    # Congelar as camadas base inicialmente
    modelo_base.trainable = False
    
    # Adicionar camadas customizadas
    entradas = tf.keras.Input(shape=(*TAMANHO_IMG, 3))
    x = modelo_base(entradas, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    saidas = Dense(NUM_RACAS, activation='softmax')(x)
    
    modelo = Model(entradas, saidas)
    
    return modelo, modelo_base

modelo, modelo_base = criar_modelo()

# Compilar o modelo
modelo.compile(
    optimizer=Adam(learning_rate=TAXA_APRENDIZADO),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar resumo do modelo
modelo.summary()

# Callbacks
checkpoint = ModelCheckpoint(
    'melhor_modelo_raca_gatos.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

parada_precoce = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduzir_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Treinar o modelo (primeira fase - apenas camadas superiores)
print("Fase 1: Treinando apenas as camadas superiores...")
historico = modelo.fit(
    dados_treino,
    epochs=5,
    validation_data=dados_validacao,
    callbacks=[checkpoint, parada_precoce, reduzir_lr]
)

# Fine-tuning: descongelar algumas camadas do modelo base
print("\nFase 2: Ajuste fino...")
modelo_base.trainable = True

# Congelar as primeiras camadas e deixar apenas as últimas treináveis
ajuste_fino_em = len(modelo_base.layers) - 20

for camada in modelo_base.layers[:ajuste_fino_em]:
    camada.trainable = False

# Recompilar com learning rate menor
modelo.compile(
    optimizer=Adam(learning_rate=TAXA_APRENDIZADO/10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Continuar treinamento
historico_ajuste = modelo.fit(
    dados_treino,
    epochs=EPOCAS,
    initial_epoch=len(historico.history['loss']),
    validation_data=dados_validacao,
    callbacks=[checkpoint, parada_precoce, reduzir_lr]
)

# Avaliar no conjunto de teste
print("\nAvaliando no conjunto de teste...")
perda_teste, acuracia_teste = modelo.evaluate(dados_teste)
print(f"Acurácia no teste: {acuracia_teste:.4f}")
print(f"Perda no teste: {perda_teste:.4f}")

# Plotar histórico de treinamento
def plotar_historico_treinamento(historico, historico_ajuste=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Combinar históricos se houver fine-tuning
    if historico_ajuste:
        acuracia = historico.history['accuracy'] + historico_ajuste.history['accuracy']
        acuracia_val = historico.history['val_accuracy'] + historico_ajuste.history['val_accuracy']
        perda = historico.history['loss'] + historico_ajuste.history['loss']
        perda_val = historico.history['val_loss'] + historico_ajuste.history['val_loss']
    else:
        acuracia = historico.history['accuracy']
        acuracia_val = historico.history['val_accuracy']
        perda = historico.history['loss']
        perda_val = historico.history['val_loss']
    
    # Plot acurácia
    ax1.plot(acuracia, label='Treino')
    ax1.plot(acuracia_val, label='Validação')
    ax1.set_title('Acurácia do Modelo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.grid(True)
    
    # Plot perda
    ax2.plot(perda, label='Treino')
    ax2.plot(perda_val, label='Validação')
    ax2.set_title('Perda do Modelo')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('historico_treinamento.png')
    plt.show()

plotar_historico_treinamento(historico, historico_ajuste)

# Função para fazer predições em novas imagens
def prever_raca_gato(caminho_imagem, modelo):
    # Carregar e preprocessar a imagem
    img = tf.keras.preprocessing.image.load_img(caminho_imagem, target_size=TAMANHO_IMG)
    array_img = tf.keras.preprocessing.image.img_to_array(img)
    array_img = np.expand_dims(array_img, axis=0)
    array_img = array_img / 255.0
    
    # Fazer predição
    predicoes = modelo.predict(array_img)
    indice_classe_prevista = np.argmax(predicoes[0])
    classe_prevista = RACAS[indice_classe_prevista]
    confianca = predicoes[0][indice_classe_prevista]
    
    # Mostrar resultados
    print(f"\nPredição para {caminho_imagem}:")
    print(f"Raça: {classe_prevista}")
    print(f"Confiança: {confianca:.2%}")
    print("\nProbabilidades para cada raça:")
    for i, raca in enumerate(RACAS):
        print(f"{raca}: {predicoes[0][i]:.2%}")
    
    return classe_prevista, confianca

# Matriz de confusão
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Obter predições para o conjunto de teste
dados_teste.reset()
predicoes = modelo.predict(dados_teste)
y_previsto = np.argmax(predicoes, axis=1)
y_verdadeiro = dados_teste.classes

# Criar matriz de confusão
matriz_confusao = confusion_matrix(y_verdadeiro, y_previsto)

# Plotar matriz de confusão
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
            xticklabels=RACAS, yticklabels=RACAS)
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.tight_layout()
plt.savefig('matriz_confusao.png')
plt.show()

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_verdadeiro, y_previsto, target_names=RACAS))

# Salvar o modelo final
modelo.save('classificador_racas_gatos_final.h5')
print("\nModelo salvo como 'classificador_racas_gatos_final.h5'")
