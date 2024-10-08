import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import gradio as gr

# Ruta donde se guardó el modelo
load_directory = '/bustosjeiner/clasificacion-resenas'

# Cargar el modelo y el tokenizador
model = DistilBertForSequenceClassification.from_pretrained(load_directory)
tokenizer = DistilBertTokenizer.from_pretrained(load_directory)

# Función para hacer predicciones
def predict_sentiment(new_reviews):
    # Validar la entrada

    # Preprocesar y tokenizar la nueva reseña
    new_encodings = tokenizer(new_reviews, truncation=True, padding='max_length', max_length=64, return_tensors='pt')

    # Realizar la predicción
    model.eval()  # Poner el modelo en modo evaluación
    with torch.no_grad():
        outputs = model(**new_encodings)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Preparar resultados para mostrar en la interfaz
    results = []
    for review, prediction in zip(new_reviews, predictions):
        sentiment = 'Positivo' if prediction.item() == 1 else 'Negativo'  # Asegúrate de usar item() para obtener el valor
        results.append(f"Reseña: {review}\nSentimiento predicho: {sentiment}")

    return "\n\n".join(results)

# Crear la interfaz de Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Escribe tu reseña aquí...", label="Reseña"),
    outputs="text",
    title="Clasificador de Sentimientos",
    description="Introduce una reseña de producto para conocer su sentimiento (positivo/negativo)."
)

# Lanzar la aplicación
iface.launch()