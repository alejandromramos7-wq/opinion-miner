import pandas as pd

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


if __name__ == "__main__":
    ARCHIVO = "opiniones_prueba.csv"

    try:
        datos = carga_base_de_datos(ARCHIVO)
        
        print("\n vista previa de los primeros 3 registros")
        print(datos.head(3))
    
    except FileNotFoundError:
        print(f"Error: No encontre el archivo '{ARCHIVO}'.")
        print("asegurate que este en la misma carpeta que este script y que el nombre este bien escrito")