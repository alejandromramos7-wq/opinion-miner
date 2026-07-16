import pandas as pd

import spacy

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Error ---> no se encontro el modelo spacy.")

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



if __name__ == "__main__":
    ARCHIVO = "opiniones_prueba.csv"

    try:
        datos = carga_base_de_datos(ARCHIVO)
        
        print("\n Limpiando y normalizando datos...")
        datos['comentario_limpio'] = datos['comentario'].apply(limpiar_texto)
        print("Limpieza completa")

        #solo es una comparativa para ver si se analizaron y limpiaron correctamente
        print("\n comparativa de textos Original vs Limpio")
        for i, fila in datos.head(3).iterrows():
            print(f"[Original]: {fila['comentario']}")
            print(f"[Limpio]: {fila['comentario_limpio']}")
    
    except FileNotFoundError:
        print(f"Error: No encontre el archivo '{ARCHIVO}'.")
        print("asegurate que este en la misma carpeta que este script y que el nombre este bien escrito")