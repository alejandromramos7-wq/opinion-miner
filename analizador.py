import pandas as pd
import spacy
from transformers import pipeline
from gensim import corpora
from gensim.models import LdaModel

# Carga el modelo spacy para la limpieza
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Error ---> no se encontro el modelo spacy.")

# Cargar el modelo de inteligencia (BETO/Robertuito)
print("cargando modelo de inteligencia (Beto)")
clasificador_ia = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis"
)
print("Modelo IA cargado correctamente")

def carga_base_de_datos(ruta_archivo):
    print("leyendo el archivo de opiniones ....")

    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo.endswith('.xlsx') or ruta_archivo.endswith('.xls'):
        df = pd.read_excel(ruta_archivo)
    else:
        raise ValueError("Formato no soportado. debe ser CSV o Excel")
    
    print("Archivo cargado con exito")
    print(f"Total de opciones encontradas: {len(df)}")
    print(f"Columnas disponibles en tu archivo: {list(df.columns)}")

    return df

def limpiar_texto(texto):
    # Metemos el resultado a la variable doc despues de haber hecho todo el texto en minusculas
    doc = nlp(str(texto).lower())

    # Agregamos una lista vacia donde caeran las palabras limpias por un bucle
    palabras_limpias = []

    for token in doc:
        # Filtros: stop words, puntuacion y espacios
        if not token.is_stop and not token.is_punct and not token.is_space:
            # Reduccion a la raiz (lematizacion)
            palabras_limpias.append(token.lemma_)

    # IMPORTANTE: Para Gensim/LDA necesitamos devolver la LISTA de palabras, no el texto unido
    return palabras_limpias

def analizar_sentimiento_ia(texto):
    # Analizar con el modelo deep learning
    resultado = clasificador_ia(str(texto)[:512])[0]
    etiqueta = resultado['label']

    traduccion = {
        'POS': 'Positivo',
        'NEG': 'Negativo',
        'NEU': 'Neutro'
    }

    return traduccion.get(etiqueta, 'Neutro')

def descubrir_temas(textos_limpios, num_temas=2):
    # Crea un diccionario con palabras unicas 
    diccionario = corpora.Dictionary(textos_limpios)
    
    # Convierte a Bolsa de Palabras (Bag of Words)
    corpus = [diccionario.doc2bow(texto) for texto in textos_limpios]

    print(f"Entrenando modelos de temas (Buscando {num_temas} temas principales)....")

    lda = LdaModel(
        corpus=corpus,
        id2word=diccionario,
        num_topics=num_temas,
        random_state=42,
        passes=10
    )

    return lda, diccionario, corpus

if __name__ == "__main__":
    ARCHIVO = "opiniones_prueba.csv"

    try:
        datos = carga_base_de_datos(ARCHIVO)
        
        print("\n Limpiando y normalizando datos...")
        # Ahora esta columna guardará una LISTA de palabras por cada fila
        datos['comentario_limpio_lista'] = datos['comentario'].apply(limpiar_texto)
        print("Limpieza completa")

        print("\n Analizando sentimiento con IA....")
        datos['sentimiento'] = datos['comentario'].apply(analizar_sentimiento_ia)
        print('---- Analisis de sentimiento completado ----')

        print("iniciando modelado de temas...")
        # Pasamos la columna correcta convertida en lista de Python
        modelo_lda, dicc, corp = descubrir_temas(datos['comentario_limpio_lista'].tolist(), num_temas=2)
        print("Deteccion de temas completada")

        print("\n [RESULTADOS SENTIMIENTOS]:")
        print("=====================================================")
        for i, fila in datos.iterrows():
            print(f"opinion {fila['id']} [{fila['sentimiento']}]: '{fila['comentario']}'")

        # Mostramos palabras claves
        print("\n [TEMAS DETECTADOS] (palabras claves mas importantes)")
        print("=====================================================")
        for idx, tema in modelo_lda.print_topics(-1):
            print(f"Tema #{idx + 1}: {tema}")
    
    except FileNotFoundError:
        print(f"Error: No encontre el archivo '{ARCHIVO}'.")
        print("asegurate que este en la misma carpeta que este script y que el nombre este bien escrito")