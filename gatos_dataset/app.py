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

RACAS = ['maine_coon', 'persa', 'siames', 'sphynx']
NUM_RACAS = len(RACAS)

EMOJIS_RACAS = {
    'maine_coon': '🦁',
    'persa': '😸',
    'siames': '😼',
    'sphynx': '🐈'
}

# Callback personalizado para atualizar o Streamlit durante o treinamento
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, metrics_placeholder, epoch_placeholder, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.metrics_placeholder = metrics_placeholder
        self.epoch_placeholder = epoch_placeholder
        self.total_epochs = total_epochs
        self.history_data = {
            'epoca': [],
            'perda': [],
            'acuracia': [],
            'perda_val': [],
            'acuracia_val': []
        }
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Atualizar progresso
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        
        # Atualizar época atual
        self.epoch_placeholder.markdown(f"### Época {epoch + 1}/{self.total_epochs}")
        
        # Guardar dados do histórico
        self.history_data['epoca'].append(epoch + 1)
        self.history_data['perda'].append(logs.get('loss', 0))
        self.history_data['acuracia'].append(logs.get('accuracy', 0))
        self.history_data['perda_val'].append(logs.get('val_loss', 0))
        self.history_data['acuracia_val'].append(logs.get('val_accuracy', 0))
        
        # Atualizar métricas na interface
        col1, col2, col3, col4 = self.metrics_placeholder.columns(4)
        col1.metric("Perda", f"{logs.get('loss', 0):.4f}")
        col2.metric("Acurácia", f"{logs.get('accuracy', 0):.4f}")
        col3.metric("Perda Val", f"{logs.get('val_loss', 0):.4f}")
        col4.metric("Acurácia Val", f"{logs.get('val_accuracy', 0):.4f}")

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

def calcular_metricas_avaliacao(modelo, dados_teste):
    """Calcula métricas detalhadas do modelo"""
    # Fazer predições
    y_pred = modelo.predict(dados_teste, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Obter labels verdadeiros
    y_true = dados_teste.classes
    
    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Relatório de classificação
    report = classification_report(y_true, y_pred_classes, 
                                 target_names=RACAS, output_dict=True)
    
    return cm, report, y_pred, y_true

# Abas principais
aba1, aba2, aba3 = st.tabs(["📊 Treinamento", "🔮 Predição", "📈 Análise"])

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
                
                try:
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
                        
                        dados_teste = gerador_teste.flow_from_directory(
                            diretorio_teste,
                            target_size=TAMANHO_IMG,
                            batch_size=TAMANHO_LOTE,
                            class_mode='categorical',
                            shuffle=False
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
                        mode='max',
                        verbose=1
                    )
                    
                    parada_precoce = EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True,
                        verbose=1
                    )
                    
                    # Placeholders para métricas
                    espaco_epoca = st.empty()
                    espaco_metricas = st.empty()
                    barra_progresso = st.progress(0)
                    
                    # Callback personalizado para Streamlit
                    callback_streamlit = StreamlitCallback(
                        barra_progresso, espaco_metricas, espaco_epoca, EPOCAS
                    )
                    
                    st.info("🏃 Treinamento em andamento...")
                    
                    # TREINAMENTO REAL
                    historico = modelo.fit(
                        dados_treino,
                        epochs=EPOCAS,
                        validation_data=dados_validacao,
                        callbacks=[checkpoint, parada_precoce, callback_streamlit],
                        verbose=0  # Silenciar saída padrão do Keras
                    )
                    
                    st.success("✅ Treinamento concluído!")
                    
                    # Salvar histórico no session state
                    st.session_state['historico_treinamento'] = callback_streamlit.history_data
                    st.session_state['modelo_treinado'] = True
                    st.session_state['modelo'] = modelo
                    st.session_state['dados_teste'] = dados_teste
                    
                    # Plotar gráficos do histórico real
                    st.subheader("📈 Histórico de Treinamento")
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Acurácia
                    ax1.plot(callback_streamlit.history_data['epoca'], 
                            callback_streamlit.history_data['acuracia'], label='Treino')
                    ax1.plot(callback_streamlit.history_data['epoca'], 
                            callback_streamlit.history_data['acuracia_val'], label='Validação')
                    ax1.set_xlabel('Época')
                    ax1.set_ylabel('Acurácia')
                    ax1.set_title('Acurácia por Época')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Perda
                    ax2.plot(callback_streamlit.history_data['epoca'], 
                            callback_streamlit.history_data['perda'], label='Treino')
                    ax2.plot(callback_streamlit.history_data['epoca'], 
                            callback_streamlit.history_data['perda_val'], label='Validação')
                    ax2.set_xlabel('Época')
                    ax2.set_ylabel('Perda')
                    ax2.set_title('Perda por Época')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    
                    # Salvar modelo
                    modelo.save('classificador_racas_gatos_streamlit.h5')
                    
                    # Avaliar no conjunto de teste
                    with st.spinner("Avaliando modelo no conjunto de teste..."):
                        avaliacao = modelo.evaluate(dados_teste, verbose=0)
                        st.session_state['avaliacao_teste'] = {
                            'perda': avaliacao[0],
                            'acuracia': avaliacao[1]
                        }
                        
                        # Calcular métricas detalhadas
                        cm, report, y_pred, y_true = calcular_metricas_avaliacao(modelo, dados_teste)
                        st.session_state['matriz_confusao'] = cm
                        st.session_state['relatorio_classificacao'] = report
                    
                    st.success(f"🎯 Acurácia no teste: {avaliacao[1]:.1%}")
                    
                except Exception as e:
                    st.error(f"❌ Erro durante o treinamento: {str(e)}")
                    st.exception(e)
                    
            else:
                st.error("Por favor, verifique os diretórios de dados!")

with aba2:
    st.header("Predição de Raça")
    
    # Verificar se existe modelo treinado
    caminho_modelo = 'classificador_racas_gatos_streamlit.h5'
    
    modelo = None
    if 'modelo' in st.session_state:
        modelo = st.session_state['modelo']
        st.success("✅ Modelo da sessão carregado!")
    elif os.path.exists(caminho_modelo):
        # Carregar modelo
        @st.cache_resource
        def carregar_modelo():
            return tf.keras.models.load_model(caminho_modelo)
        
        modelo = carregar_modelo()
        st.success("✅ Modelo salvo carregado!")
    
    if modelo is not None:
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
                
                # Verificar se a imagem tem 3 canais (RGB)
                if len(array_img.shape) == 2:  # Imagem em escala de cinza
                    array_img = np.stack([array_img] * 3, axis=-1)
                elif array_img.shape[2] == 4:  # Imagem RGBA
                    array_img = array_img[:, :, :3]
                
                array_img = np.expand_dims(array_img, axis=0)
                array_img = array_img / 255.0
                
                # Fazer predição REAL
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
                
                # Criar DataFrame com resultados REAIS
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

with aba3:
    st.header("Análise do Modelo")
    
    if st.session_state.get('modelo_treinado', False) and 'avaliacao_teste' in st.session_state:
        
        # Métricas REAIS
        st.subheader("📊 Métricas de Performance")
        
        avaliacao = st.session_state['avaliacao_teste']
        relatorio = st.session_state.get('relatorio_classificacao', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Valores REAIS do modelo
        acuracia_final = avaliacao['acuracia']
        perda_final = avaliacao['perda']
        
        # Calcular precisão e recall médios
        if relatorio and 'macro avg' in relatorio:
            precisao_media = relatorio['macro avg']['precision']
            recall_medio = relatorio['macro avg']['recall']
        else:
            precisao_media = acuracia_final  
            recall_medio = acuracia_final
        
        col1.metric("Acurácia Final", f"{acuracia_final:.1%}")
        col2.metric("Perda Final", f"{perda_final:.3f}")
        col3.metric("Precisão Média", f"{precisao_media:.1%}")
        col4.metric("Recall Médio", f"{recall_medio:.1%}")
        
        # Matriz de Confusão REAL
        if 'matriz_confusao' in st.session_state:
            st.subheader("🎯 Matriz de Confusão")
            
            matriz_confusao = st.session_state['matriz_confusao']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues',
                        xticklabels=RACAS, yticklabels=RACAS, ax=ax)
            ax.set_title('Matriz de Confusão')
            ax.set_xlabel('Predito')
            ax.set_ylabel('Verdadeiro')
            st.pyplot(fig)
        
        # Métricas por classe REAIS
        if relatorio:
            st.subheader("📈 Performance por Raça")
            
            dados_metricas = {
                'Raça': [f"{EMOJIS_RACAS[raca]} {raca.replace('_', ' ').title()}" for raca in RACAS],
                'Precisão': [relatorio.get(raca, {}).get('precision', 0) for raca in RACAS],
                'Recall': [relatorio.get(raca, {}).get('recall', 0) for raca in RACAS],
                'F1-Score': [relatorio.get(raca, {}).get('f1-score', 0) for raca in RACAS]
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
            
            # Gráfico radar com dados REAIS
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
            
        # Mostrar histórico de treinamento REAL
        if 'historico_treinamento' in st.session_state:
            st.subheader("📈 Histórico de Treinamento Detalhado")
            
            historico = st.session_state['historico_treinamento']
            
            # Criar gráficos do histórico
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Acurácia
            ax1.plot(historico['epoca'], historico['acuracia'], label='Treino', marker='o')
            ax1.plot(historico['epoca'], historico['acuracia_val'], label='Validação', marker='s')
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Acurácia')
            ax1.set_title('Acurácia por Época')
            ax1.legend()
            ax1.grid(True)
            
            # Perda
            ax2.plot(historico['epoca'], historico['perda'], label='Treino', marker='o')
            ax2.plot(historico['epoca'], historico['perda_val'], label='Validação', marker='s')
            ax2.set_xlabel('Época')
            ax2.set_ylabel('Perda')
            ax2.set_title('Perda por Época')
            ax2.legend()
            ax2.grid(True)
            
            # Diferença entre treino e validação (Overfitting)
            diff_acc = np.array(historico['acuracia']) - np.array(historico['acuracia_val'])
            ax3.plot(historico['epoca'], diff_acc, marker='o', color='red')
            ax3.set_xlabel('Época')
            ax3.set_ylabel('Diferença de Acurácia')
            ax3.set_title('Overfitting (Treino - Validação)')
            ax3.grid(True)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            ax4.bar(range(len(RACAS)), [1]*len(RACAS), color=['blue', 'green', 'red', 'orange'])
            ax4.set_xlabel('Raças')
            ax4.set_ylabel('Normalizado')
            ax4.set_title('Distribuição de Classes')
            ax4.set_xticks(range(len(RACAS)))
            ax4.set_xticklabels(RACAS, rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)
        
    else:
        st.warning("⚠️ Nenhum modelo treinado encontrado. Por favor, treine um modelo primeiro.")