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
