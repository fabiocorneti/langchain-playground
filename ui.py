"""
Simple UI for search API
"""
import json
import os
from urllib.parse import quote
import pandas
import requests
import streamlit as st


def ask(question: str) -> pandas.DataFrame:
    """
    Send a question.
    """
    api_url = "http://host.docker.internal:8000/ask"
    response = requests.post(api_url, json.dumps({
        "question": question
    }), timeout=300).json()
    answer = response["answer"]
    st.write("### Answer")
    st.write(f"{answer}\n\n---")
    st.write("### Snippets")
    dataframe = pandas.json_normalize(response["results"])
    for _, row in dataframe.reset_index().iterrows():
        source = row['metadata.source'].strip()
        title = row['metadata.title']
        if source.startswith("http://") or source.startswith("https://"):
            if title:
                source_repr = f"[{title}]({source})"
            else:
                source_repr = source
        else:
            source_repr = quote(os.path.basename(source))
        page = row.get('metadata.page', None)
        score = row.get('score')
        if page is not None and page > 0:
            st.markdown(f"**{source_repr} - Page: {page} - Score: {score}**")
        else:
            st.markdown(f"**{source_repr} - Score: {score}**")
        st.caption(row["page_content"])
        if page:
            st.markdown(f"![Preview](http://localhost:8000/pdfpage?filename={source}&page={page})")


def main():
    """
    UI definition.
    """
    st.title("Search")

    question = st.text_area("Question")
    if st.button("Ask"):
        ask(question)


if __name__ == '__main__':
    main()
