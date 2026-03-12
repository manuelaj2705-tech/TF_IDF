import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Buscador Inteligente de Textos", page_icon="🔎", layout="wide")

st.title("🔎 Buscador Inteligente de Información")

st.write("""
Esta herramienta permite **buscar información dentro de varios textos** usando técnicas de análisis de lenguaje natural.

📌 **¿Cómo funciona?**  
1. Escribe varios textos o frases en el cuadro de documentos.  
2. Cada línea se interpreta como un **documento independiente**.  
3. Luego escribe una **pregunta o consulta**.  
4. El sistema comparará tu pregunta con los documentos y mostrará **el más relacionado**.
""")

col1, col2 = st.columns([2,1])

with col1:

    text_input = st.text_area(
        "📄 Base de textos para analizar (uno por línea):",
        "Los estudiantes estudian programación en la universidad.\n"
        "La inteligencia artificial ayuda a resolver problemas complejos.\n"
        "El ejercicio físico mejora la salud y el bienestar.\n"
        "La música puede cambiar el estado de ánimo de las personas.\n"
        "Los viajes permiten conocer nuevas culturas."
    )

    question = st.text_input("💬 Escribe una consulta sobre los textos:")

    if st.button("⚡ Buscar información"):

        if text_input.strip() == "" or question.strip() == "":
            st.warning("Por favor escribe algunos textos y una consulta.")
        else:

            documents = text_input.split("\n")

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(documents + [question])

            similarity = cosine_similarity(
                tfidf_matrix[-1],
                tfidf_matrix[:-1]
            )

            best_match_index = similarity.argmax()
            best_document = documents[best_match_index]
            score = similarity[0][best_match_index]

            st.subheader("📊 Resultado del análisis")

            st.write("El texto más relacionado con tu consulta es:")

            st.success(best_document)

            st.write(f"Nivel de coincidencia: **{score:.2f}**")

with col2:

    st.subheader("💡 Ejemplos de consultas")

    st.write("¿Qué ayuda a resolver problemas complejos?")
    st.write("¿Qué actividad mejora la salud?")
    st.write("¿Qué puede cambiar el estado de ánimo?")
    st.write("¿Para qué sirven los viajes?")
