# HSE Criminal Cases Investigation

**by Lebedyuk Eva, Vdonin Aleksei**  
Faculty of Computer Science, HSE University  

## Coursework 2024  
This coursework focuses on the development of a system for **Criminal Cases Investigation** using **Named Entity Recognition (NER)** to extract key entities from judicial texts.

### Overview  
The project aims to automate the analysis of judicial documents, specifically focusing on **Article 105, Part 1 of the Russian Criminal Code ("Murder")**. It includes the following key components:
1. **NER Model**: Fine-tuned a BERT-based model for extracting entities such as individuals, legal entities, crimes, laws, and penalties.
2. **Summarization Module**: Implemented extractive summarization for condensing large judicial texts by more than 90% while preserving key information.
3. **Web Application**: Developed an interactive interface using **Streamlit** for entity extraction and summarization, enabling efficient document analysis.

### Key Features  
- **NER Pipeline**: Automatically identifies and categorizes entities in legal texts.
- **Text Summarization**: Provides concise summaries of judicial documents for faster comprehension.
- **Interactive Interface**: Allows users to input texts, visualize predictions, and interact with summarized outputs.
- **Optimized Deployment**: Application optimized for CPU usage, ensuring accessibility on standard systems.

### Contributions  
- **Lebedyuk Eva**: Led the development of the NER model, including data preprocessing, training, and evaluation.
- **Vdonin Aleksei**: Developed the web application interface and implemented the summarization module.

### Project Details  
- **Dataset**: The model was trained on the **RuLegalNER** dataset, adapted for extracting legal entities in Russian texts.
- **Tech Stack**:
  - Python (Hugging Face Transformers, Streamlit)
  - Libraries: BeautifulSoup, PyMyStem, JSON
  - Infrastructure: Google Colab, Streamlit Cloud

### Challenges  
- High computational demands for model training.
- Integration of multiple components into a seamless application.

### Future Work  
- Enhance entity recall for legal entities and penalties.
- Expand summarization functionality with hybrid models.
- Integrate additional entity types, such as dates and participant relationships.

### Link to the Project Web App:  
[HSE Criminal Cases Web App](https://hse-criminal-cases.streamlit.app)
