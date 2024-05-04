# intellihack_team_vertex_task_3

## Chatbot Project

This project implements a chatbot using LangChain and ChainLit. The chatbot is designed to answer questions by retrieving information from a FAISS vector store, powered by a backend of OpenAI's GPT-3.5 Turbo model. This setup allows for accurate and fast responses suitable for various use cases including but not limited to customer service and information retrieval.

## Getting Started

These instructions will guide you through setting up your project locally. By following these steps, you can get a copy of the project running on your local machine for development and testing purposes.

### Installation

#### Step 1: Clone the Repository

Clone this repository to your local machine to get started with the chatbot project.

```bash
git clone https://github.com/Tharindu209-playground/intellihack_team_vertex_task_3.git
cd intellihack_team_vertex_task_3
```

#### Step 2: Set Up a Virtual Environment

Set up a Python virtual environment to manage dependencies separately from your global Python setup.

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On MacOS/Linux
source venv/bin/activate
```

#### Step 3: Install Dependencies

Install the necessary Python packages specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Running the Application

To run the application, use the following command:

```bash
chainlit run model.py
```

This will start the server, and you can interact with the chatbot via the web interface provided by ChainLit.

## Usage

Once the application is running, navigate to the provided local URL. You can start interacting with the chatbot by typing your questions into the web interface. The chatbot will respond based on the information retrieved from the configured data sources.
