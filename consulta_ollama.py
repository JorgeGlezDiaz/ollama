import requests
import json
import os

# Definir la URL del endpoint de la API local de Ollama
url = "http://localhost:11434/api/generate"

# Definir el cuerpo de la solicitud
data = {
    "model": "qwen2.5:0.5b",  # Asegúrate de que el nombre del modelo sea correcto
    "prompt": "My name is Jorge.",
    "stream": False
}

# Definir los encabezados de la solicitud
headers = {
    "Content-Type": "application/json"
}

# Enviar la solicitud POST a la API
response = requests.post(url, headers=headers, data=json.dumps(data))

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    # Convertir la respuesta a JSON
    resultado = response.json()

    # Nombre del archivo donde se guardarán todas las respuestas
    file_name = "respuestas_acumuladas.json"

    # Si el archivo existe, cargar los datos previos; si no, crear un nuevo array
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as file:
            respuestas = json.load(file)
    else:
        respuestas = []

    # Agregar la nueva respuesta al array
    respuestas.append(resultado)

    # Guardar el array actualizado en el archivo
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(respuestas, file, indent=4, ensure_ascii=False)

    # Mostrar solo la respuesta en la consola
    print(resultado["response"])

else:
    print(f"Error en la solicitud: {response.status_code}")
    print(response.text)


### Obtener info del JSON

with open("respuestas_acumuladas.json", "r", encoding="utf-8") as file:
    datos = json.load(file)

print(datos)



# Leer el archivo y cargar los datos
try:
    with open(file_name, "r", encoding="utf-8") as file:
        respuestas = json.load(file)  # Carga todas las respuestas en una lista

    # Extraer y mostrar el tiempo de carga del modelo de cada respuesta
    for i, respuesta in enumerate(respuestas, start=1):
        if "response" in respuesta:
            print(f"La respuesta {i} fue de {respuesta['response']}.")
        else:
            print(f"Respuesta {i} - No se encontró 'response' en esta respuesta")
        if "load_duration" in respuesta:
            print(f"tuvo un tiempo de carga de {respuesta['load_duration']} nanosegundos")
        else:
            print(f"Respuesta {i} - No se encontró 'load_duration' en esta respuesta")
        if "eval_count" in respuesta:
            print(f"El número de tokens generados en la respuesta {i} fue de {respuesta['eval_count']}.")
        else:
            print(f"Respuesta {i} - No se encontró 'eval_count' en esta respuesta")

except FileNotFoundError:
    print(f"El archivo '{file_name}' no existe.")
except json.JSONDecodeError:
    print(f"Error al leer el archivo JSON. Verifica su formato.")