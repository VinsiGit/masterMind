# Dishify Setup Instructions

Follow these steps to set up and run the Dishify application:

### Step 1: Create a Hugging Face Token
- Generate a Hugging Face token.
- Add the token to a `.env` file in your project directory.

### Step 2: Configure Your Server
- Set up a website for the server to run on and include it in the `.env` file.
- If you don't have a server, you can use `localhost`.

### Step 3: Build and Run the Application
Run the following command to build and start the Docker container:
```bash
docker compose up --build -d
```

### Step 4: Install the Llama 3.2:3B Model
To install the Llama 3.2:3B model in the Docker container, execute:

```bash
docker exec -it dishify-ollama ollama pull llama3.2:3b
```