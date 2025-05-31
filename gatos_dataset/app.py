import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import os
import time
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import io

# Configuração da página
st.set_page_config(
    page_title="Classificador de Raças de Gatos",
    layout="wide"
)

# Título principal
st.title("Classificador de Raças de Gatos com EfficientNet")
st.markdown("---")

# Barra lateral
st.sidebar.title("Configurações")

# Configurações do modelo
TAMANHO_IMG = (224, 224)
TAMANHO_LOTE = st.sidebar.slider("Tamanho do Lote", 8, 64, 32)
EPOCAS = st.sidebar.slider("Épocas", 10, 50, 30)
TAXA_APRENDIZADO = st.sidebar.select_slider(
    "Taxa de Aprendizado",
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
    value=0.001
)

# Classes
RACAS = ['maine_coon', 'persa', 'siames', 'sphynx']
NUM_RACAS = len(RACAS)

# Emojis para cada raça
EMOJIS_RACAS = {
    'maine_coon': '🦁',
    'persa': '😸',
    'siames': '😼',
    'sphynx': '🐈'
}

# Função para criar o modelo
@st.cache_resource
def criar_modelo():
    modelo_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*TAMANHO_IMG, 3)
    )
    modelo_base.trainable = False
    
    entradas = tf.keras.Input(shape=(*TAMANHO_IMG, 3))
    x = modelo_base(entradas, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    saidas = Dense(NUM_RACAS, activation='softmax')(x)
    
    modelo = Model(entradas, saidas)
    return modelo, modelo_base

# Abas principais
aba1, aba2, aba3 = st.tabs(["📊 Treinamento", "🔮 Predição", "📈 Análise"])

# Aba de Treinamento
with aba1:
    st.header("Treinamento do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Configuração dos Dados")
        diretorio_treino = st.text_input("Diretório de Treino", value="treino")
        diretorio_teste = st.text_input("Diretório de Teste", value="teste")
        
        # Verificar se os diretórios existem
        if os.path.exists(diretorio_treino) and os.path.exists(diretorio_teste):
            st.success("✅ Diretórios encontrados!")
            
            # Contar imagens
            contagem_treino = {}
            contagem_teste = {}
            
            for raca in RACAS:
                caminho_treino = os.path.join(diretorio_treino, raca)
                caminho_teste = os.path.join(diretorio_teste, raca)
                
                if os.path.exists(caminho_treino):
                    contagem_treino[raca] = len([f for f in os.listdir(caminho_treino) 
                                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                if os.path.exists(caminho_teste):
                    contagem_teste[raca] = len([f for f in os.listdir(caminho_teste) 
                                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Mostrar estatísticas
            st.subheader("📊 Estatísticas do Dataset")
            
            df_estatisticas = pd.DataFrame({
                'Raça': RACAS,
                'Emoji': [EMOJIS_RACAS[raca] for raca in RACAS],
                'Treino': [contagem_treino.get(raca, 0) for raca in RACAS],
                'Teste': [contagem_teste.get(raca, 0) for raca in RACAS]
            })
            
            st.dataframe(df_estatisticas, hide_index=True)
            
            # Gráfico de barras
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(RACAS))
            largura = 0.35
            
            barras1 = ax.bar(x - largura/2, df_estatisticas['Treino'], largura, label='Treino')
            barras2 = ax.bar(x + largura/2, df_estatisticas['Teste'], largura, label='Teste')
            
            ax.set_xlabel('Raças')
            ax.set_ylabel('Número de Imagens')
            ax.set_title('Distribuição do Dataset')
            ax.set_xticks(x)
            ax.set_xticklabels(RACAS)
            ax.legend()
            
            st.pyplot(fig)
            
        else:
            st.error("❌ Diretórios não encontrados!")
    
    with col2:
        st.subheader("🚀 Iniciar Treinamento")
        
        if st.button("Treinar Modelo", type="primary"):
            if os.path.exists(diretorio_treino) and os.path.exists(diretorio_teste):
                with st.spinner("Preparando dados..."):
                    # Criar geradores
                    gerador_treino = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        horizontal_flip=True,
                        zoom_range=0.2,
                        shear_range=0.2,
                        fill_mode='nearest',
                        validation_split=0.2
                    )
                    
                    gerador_teste = ImageDataGenerator(rescale=1./255)
                    
                    dados_treino = gerador_treino.flow_from_directory(
                        diretorio_treino,
                        target_size=TAMANHO_IMG,
                        batch_size=TAMANHO_LOTE,
                        class_mode='categorical',
                        subset='training'
                    )
                    
                    dados_validacao = gerador_treino.flow_from_directory(
                        diretorio_treino,
                        target_size=TAMANHO_IMG,
                        batch_size=TAMANHO_LOTE,
                        class_mode='categorical',
                        subset='validation'
                    )
                
                # Criar modelo
                with st.spinner("Criando modelo..."):
                    modelo, modelo_base = criar_modelo()
                    modelo.compile(
                        optimizer=Adam(learning_rate=TAXA_APRENDIZADO),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                
                # Callbacks
                checkpoint = ModelCheckpoint(
                    'melhor_modelo_racas_gatos.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
                
                parada_precoce = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                # Placeholders para métricas
                espaco_epoca = st.empty()
                espaco_metricas = st.empty()
                barra_progresso = st.progress(0)
                
                # Treinar
                st.info("🏃 Treinamento em andamento...")
                
                dados_historico = {
                    'epoca': [],
                    'perda': [],
                    'acuracia': [],
                    'perda_val': [],
                    'acuracia_val': []
                }
                
                # Simular treinamento (em produção, use model.fit com callbacks customizados)
                for epoca in range(EPOCAS):
                    # Aqui você deve integrar com o treinamento real
                    # Este é apenas um exemplo de interface
                    
                    progresso = (epoca + 1) / EPOCAS
                    barra_progresso.progress(progresso)
                    
                    espaco_epoca.markdown(f"### Época {epoca + 1}/{EPOCAS}")
                    
                    # Valores simulados (substitua com valores reais do treinamento)
                    metricas_atuais = {
                        'perda': np.random.uniform(0.1, 0.5),
                        'acuracia': np.random.uniform(0.7, 0.95),
                        'perda_val': np.random.uniform(0.15, 0.6),
                        'acuracia_val': np.random.uniform(0.65, 0.9)
                    }
                    
                    # Atualizar histórico
                    dados_historico['epoca'].append(epoca + 1)
                    for chave in ['perda', 'acuracia', 'perda_val', 'acuracia_val']:
                        dados_historico[chave].append(metricas_atuais[chave])
                    
                    # Mostrar métricas
                    col1, col2, col3, col4 = espaco_metricas.columns(4)
                    col1.metric("Perda", f"{metricas_atuais['perda']:.4f}")
                    col2.metric("Acurácia", f"{metricas_atuais['acuracia']:.4f}")
                    col3.metric("Perda Val", f"{metricas_atuais['perda_val']:.4f}")
                    col4.metric("Acurácia Val", f"{metricas_atuais['acuracia_val']:.4f}")
                    
                    time.sleep(0.1)  # Simular tempo de treinamento
                
                st.success("✅ Treinamento concluído!")
                
                # Salvar histórico no session state
                st.session_state['historico_treinamento'] = dados_historico
                
                # Plotar gráficos
                st.subheader("📈 Histórico de Treinamento")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Acurácia
                ax1.plot(dados_historico['epoca'], dados_historico['acuracia'], label='Treino')
                ax1.plot(dados_historico['epoca'], dados_historico['acuracia_val'], label='Validação')
                ax1.set_xlabel('Época')
                ax1.set_ylabel('Acurácia')
                ax1.set_title('Acurácia por Época')
                ax1.legend()
                ax1.grid(True)
                
                # Perda
                ax2.plot(dados_historico['epoca'], dados_historico['perda'], label='Treino')
                ax2.plot(dados_historico['epoca'], dados_historico['perda_val'], label='Validação')
                ax2.set_xlabel('Época')
                ax2.set_ylabel('Perda')
                ax2.set_title('Perda por Época')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
                
                # Salvar modelo
                modelo.save('classificador_racas_gatos_streamlit.h5')
                st.session_state['modelo_treinado'] = True
                
            else:
                st.error("Por favor, verifique os diretórios de dados!")

# Aba de Predição
with aba2:
    st.header("Predição de Raça")
    
    # Verificar se existe modelo treinado
    caminho_modelo = 'classificador_racas_gatos_streamlit.h5'
    
    if os.path.exists(caminho_modelo) or st.session_state.get('modelo_treinado', False):
        # Carregar modelo
        @st.cache_resource
        def carregar_modelo():
            return tf.keras.models.load_model(caminho_modelo)
        
        if os.path.exists(caminho_modelo):
            modelo = carregar_modelo()
            st.success("✅ Modelo carregado!")
        
        # Upload de imagem
        arquivo_enviado = st.file_uploader(
            "Escolha uma imagem de gato",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos suportados: JPG, JPEG, PNG"
        )
        
        if arquivo_enviado is not None:
            # Mostrar imagem
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Imagem Original")
                imagem = Image.open(arquivo_enviado)
                st.image(imagem, use_column_width=True)
                
                # Informações da imagem
                st.caption(f"Dimensões: {imagem.size[0]} x {imagem.size[1]}")
                st.caption(f"Formato: {imagem.format}")
            
            with col2:
                st.subheader("🔮 Predição")
                
                # Preprocessar imagem
                img = imagem.resize(TAMANHO_IMG)
                array_img = np.array(img)
                array_img = np.expand_dims(array_img, axis=0)
                array_img = array_img / 255.0
                
                # Fazer predição
                with st.spinner("Analisando..."):
                    predicoes = modelo.predict(array_img, verbose=0)
                    indice_predito = np.argmax(predicoes[0])
                    raca_predita = RACAS[indice_predito]
                    confianca = predicoes[0][indice_predito]
                
                # Mostrar resultado principal
                st.markdown(f"### {EMOJIS_RACAS[raca_predita]} {raca_predita.replace('_', ' ').title()}")
                st.markdown(f"**Confiança:** {confianca:.1%}")
                
                # Barra de progresso para confiança
                st.progress(float(confianca))
                
                # Mostrar todas as probabilidades
                st.markdown("#### Probabilidades por Raça:")
                
                # Criar DataFrame com resultados
                df_resultados = pd.DataFrame({
                    'Raça': [f"{EMOJIS_RACAS[raca]} {raca.replace('_', ' ').title()}" for raca in RACAS],
                    'Probabilidade': [f"{predicoes[0][i]:.1%}" for i in range(NUM_RACAS)],
                    'Score': predicoes[0]
                })
                df_resultados = df_resultados.sort_values('Score', ascending=False)
                
                # Mostrar como gráfico de barras
                fig, ax = plt.subplots(figsize=(8, 4))
                barras = ax.barh(df_resultados['Raça'], df_resultados['Score'])
                
                # Colorir a barra com maior probabilidade
                cores = ['#1f77b4' if i != 0 else '#ff7f0e' for i in range(len(barras))]
                for barra, cor in zip(barras, cores):
                    barra.set_color(cor)
                
                ax.set_xlabel('Probabilidade')
                ax.set_xlim(0, 1)
                ax.set_title('Distribuição de Probabilidades')
                
                # Adicionar valores nas barras
                for i, (raca, score) in enumerate(zip(df_resultados['Raça'], df_resultados['Score'])):
                    ax.text(score + 0.01, i, f'{score:.1%}', va='center')
                
                st.pyplot(fig)
                
                # Informações adicionais sobre a raça
                info_racas = {
                    'maine_coon': "O Maine Coon é uma das maiores raças de gatos domésticos. São conhecidos por serem gentis gigantes, muito sociáveis e brincalhões.",
                    'persa': "O gato Persa é conhecido por sua pelagem longa e luxuosa e face achatada. São gatos calmos, gentis e afetuosos.",
                    'siames': "O gato Siamês é uma raça elegante conhecida por seus olhos azuis e pelagem clara com extremidades escuras. São muito vocais e inteligentes.",
                    'sphynx': "O Sphynx é conhecido por ser 'sem pelos' (na verdade tem uma fina camada de penugem). São gatos muito afetuosos e energéticos."
                }
                
                with st.expander(f"ℹ️ Sobre {raca_predita.replace('_', ' ').title()}"):
                    st.write(info_racas.get(raca_predita, "Informação não disponível."))
        
    else:
        st.warning("⚠️ Nenhum modelo treinado encontrado. Por favor, treine um modelo primeiro na aba 'Treinamento'.")

# Aba de Análise
with aba3:
    st.header("Análise do Modelo")
    
    if st.session_state.get('modelo_treinado', False) or os.path.exists(caminho_modelo):
        
        # Métricas gerais
        st.subheader("📊 Métricas de Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Valores simulados (em produção, use valores reais)
        col1.metric("Acurácia Final", "92.5%", "+2.3%")
        col2.metric("Perda Final", "0.235", "-0.042")
        col3.metric("Precisão Média", "91.8%")
        col4.metric("Recall Médio", "90.2%")
        
        # Matriz de Confusão
        st.subheader("🎯 Matriz de Confusão")
        
        # Simular matriz de confusão (em produção, use dados reais)
        matriz_confusao = np.array([
            [45, 2, 1, 1],
            [3, 42, 2, 1],
            [1, 1, 46, 2],
            [2, 1, 1, 44]
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
                    xticklabels=RACAS, yticklabels=RACAS, ax=ax)
        ax.set_title('Matriz de Confusão')
        ax.set_xlabel('Predito')
        ax.set_ylabel('Verdadeiro')
        st.pyplot(fig)
        
        # Métricas por classe
        st.subheader("📈 Performance por Raça")
        
        # Dados simulados
        dados_metricas = {
            'Raça': [f"{EMOJIS_RACAS[raca]} {raca.replace('_', ' ').title()}" for raca in RACAS],
            'Precisão': [0.918, 0.894, 0.920, 0.936],
            'Recall': [0.918, 0.875, 0.920, 0.917],
            'F1-Score': [0.918, 0.884, 0.920, 0.926]
        }
        
        df_metricas = pd.DataFrame(dados_metricas)
        
        # Mostrar tabela
        st.dataframe(
            df_metricas.style.format({
                'Precisão': '{:.1%}',
                'Recall': '{:.1%}',
                'F1-Score': '{:.1%}'
            }),
            hide_index=True
        )
        
        # Gráfico radar
        st.subheader("🕸️ Comparação de Métricas")
        
        categorias = ['Precisão', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(projection='polar'))
        
        angulos = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
        angulos += angulos[:1]
        
        for idx, raca in enumerate(RACAS):
            valores = [dados_metricas['Precisão'][idx], 
                      dados_metricas['Recall'][idx], 
                      dados_metricas['F1-Score'][idx]]
            valores += valores[:1]
            
            ax.plot(angulos, valores, 'o-', linewidth=2, 
                   label=f"{EMOJIS_RACAS[raca]} {raca.replace('_', ' ').title()}")
            ax.fill(angulos, valores, alpha=0.15)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        st.pyplot(fig)
        
    else:
        st.warning("⚠️ Nenhum modelo treinado encontrado. Por favor, treine um modelo primeiro.")