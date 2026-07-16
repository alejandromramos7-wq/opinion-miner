import pandas as pd

import spacy

from transformers import pipeline


#carga el modelo spacy para la limpieza
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Error ---> no se encontro el modelo spacy.")

    #cargar le modelo de inteligencia (BETO)

print("cargando modelo de inteligencia (Beto)")
clasificador_ia = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis")
print("Modelo IA cargado correctamente")
#agrupacion cluesterin 
#clasificacion preciccion anova

def carga_base_de_datos(ruta_archivo):
    print("leyendo el archivo de opiniones ....")

    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo('.xlsx') or ruta_archivo.endswith('.xls'):
        df = pd.read_excel(ruta_archivo)
    else:
        raise ValueError("Formato no soportado. debe ser CVS o Excel")
    
    print("Archivo cargado con exito")
    print(f"Total de opciones encontradas:{len(df)}")
    print(f"Columnas disponibles en tu archivo: {list(df.columns)}")

    return df


def limpiar_texto(texto):
    #metemos el resultaod a la variable doc despues de haber hecho todo el texto en minusculas y analizado por spacy
    doc = nlp(str(texto).lower())

    #agregamos una lista vacia donde caeran las palabras limpias por un bucle
    palabras_limpias = []

    for token in doc:
        #is_stop --> todas las palabras que que no sean paradas como el, la , los, un , de , para, que (no aportan nada al analisis)
        #is_puntc --> todas las que no sean signos de puntuacion(elimina puntos, comas, signos de interrogacion. etc)
        #is_espace -->que no sean espacio blanco en blanco extra un tabulador y salto de linea
        #todos los mencionados con el not es que no los quiero 
        if not token.is_stop and not token.is_punct and not token.is_space:
            #lemma_ --> reduccion a la raiz, leatiza las palabras transformandolas a su infinitivo o cambian a su masculino singular
            palabras_limpias.append(token.lemma_)

    #agrega un epacio junto a cada palabra
    return " ".join(palabras_limpias)

def analizar_sentimiento_ia(texto):
    
    #analizar con el modelo dep learning
    resultado = clasificador_ia(str(texto)[:512])[0]

    etiqueta = resultado['label']

    traduccion = {
        'POS': 'Positivo',
        'NEG': 'Negativo',
        'NEU': 'Neutro'
    }

    return traduccion.get(etiqueta, 'Neutro')




if __name__ == "__main__":
    ARCHIVO = "opiniones_prueba.csv"

    try:
        datos = carga_base_de_datos(ARCHIVO)
        
        print("\n Limpiando y normalizando datos...")
        datos['comentario_limpio'] = datos['comentario'].apply(limpiar_texto)
        print("Limpieza completa")

        print("\n Analizando sentimiento con IA....")
        datos['sentimiento'] = datos['comentario'].apply(analizar_sentimiento_ia)
        print('---- Analisis de sentimiento completado ----')

        
        #solo es una comparativa para ver si se analizaron y limpiaron correctamente
        print("---Resultados---")
        print("========================")
        for i, fila in datos.iterrows():
            print(f"opinion {fila['id']}: [{fila['sentimiento']}]")
            print(f"'{fila['comentario']}'")
            print("-" * 50)
    
    except FileNotFoundError:
        print(f"Error: No encontre el archivo '{ARCHIVO}'.")
        print("asegurate que este en la misma carpeta que este script y que el nombre este bien escrito")

