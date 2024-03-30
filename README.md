# Updates

Updated dependencies to be current:

- jupyterlab==4.1.2
- requests==2.31.0
- dvc==3.48.1
- dvc-s3==3.0.1
- pandas==1.5.0
- pyarrow==9.0.0
- gradio==4.19.2
- torch==2.2.1
- sentence-transformers==2.5.1
- bertopic==0.16.0

# LLMSherpa

Pulled in LLMSherpa to help with parsing multiple types of documents (specifically for PDFs) into chunks best to use for RAG.

To work locally, run the docker container locally
```docker run -p 5010:5001 ghcr.io/nlmatics/nlm-ingestor:latest```

# ChromaDB

Switched the storage of the document embeddings to use chromadb instead of the custom approach we built initially. Interesting metrics:

- Abstract Embeddings stored by numpy uses 814M of storage
- Same embeddings stored in chromadb along with the text value uses 8G of storage

# spaceapps2022
Huntsville AI submission for Space Apps 2022

# First thoughts:

1. Get something that minimally meets the requirements of the challenge:
    1. Download a bunch of PDFs from the NASA server, initially throw into a /data folder here. Probably follow a kedro type layout, even if not using kedro.
    2. Figure out how to extract text from PDF - currently looking at several options for this - leading candidate is PyPDF2 since it provides an easy way to also pull images
    3. Crank up a summarization, topic mapping, word cloud, etc to create something from this word soup
    4. Spin up a streamlit to display - have streamlit host based on this repo


2. Get fancy - lots of thoughts here - feel free to throw more in
    1. Graph representation with documents and attributes and some fancy (already built & free) way to visualize and traverse
    2. Start grabbing a few videos/recordings from the same NASA server - transcribe and store text - then follow similar approach to PDF
    3. Extract images from PDFs and run object detection - add these to the list of keywords for the document.


3. Somewhere, we'll have to make a video to present - not sure how to go about that except for doing something cheap or stupid on my phone.

