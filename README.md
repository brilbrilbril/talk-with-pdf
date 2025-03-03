**Installation**
1. Install Ollama [here](https://ollama.com/)
2. Pull the embeddings model and LLM.

    For the embedding model:
    ``` ollama pull nomic-embed-text```

    For the LLM:
    ``` ollama pull llama3.2:3b```

    You can change the model or the parameter if you have better device.

3. Clone this repository
4. (Recommended) Use virtual environment from python or other library by creating venv.

    ```python -m venv venv```

    Then activate it using

    ```venv/Scripts/activate```
4. Install dependencies using ```pip install -r requirements.txt```

    ***Note***: Please tell me if there's missing dependecy because I forgot what dependencies I have installed.
5.  Run ```streamlit run app/main.py```

**How to Use**
1. Upload your PDF file.
    **Note**: You need to refresh the page if your PDF is not successfully uploaded 
2. Click the ***Generate report*** to generate simple report from your PDF.
3. Click the ***Talk*** if you want to ask question about your PDF.
    ***IF*** it can't answer correctly, try to rephrase your question.

    Example: *what is diabetes?* **->** *explain me about diabetes.*

**Weakness**
1. Rough interface design.
2. It can't answer simple question like *"what is the title?"* or *"tell me about the section A"*
3. The performance depends on device. If you are using GPU, then the the answer will generated faster. Vice versa.
4. Not all PDF can't be displayed.
