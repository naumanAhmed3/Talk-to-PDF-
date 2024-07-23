# Chat PDF Project

This application allows you to upload and interact with PDF documents using conversational AI. You can run this project using either Streamlit or Docker.

## Tools

- Python 3.8+
- Streamlit 
- LangChain
- Gemini 
- Docker 

## Getting Started

### Running with Streamlit

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Set the Gemini API key environment variable:

    **Windows:**
    ```sh
    setx GEMINI_API_KEY "YOUR_GEMINI_API_KEY"
    ```

    **macOS/Linux:**
    ```sh
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the Streamlit application:
    ```sh
    streamlit run main.py
    ```

The application will run on [http://localhost:8501](http://localhost:8501).

### Running with Docker

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Build and run the Docker image:
    ```sh
    docker build -t your-username/myproject .
    docker run -p 8501:8501 -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY" your-username/myproject
    ```

The application will run on [http://localhost:8501](http://localhost:8501).
