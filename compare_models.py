import time
import random
from textblob import TextBlob  # For basic sentiment analysis
from tabulate import tabulate  # For pretty-printing the comparison table

# Define the prompt
prompt = "What is the future of artificial intelligence?"

# Simulated model responses (in practice, you'd call actual model APIs)
def simulate_model_response(model_name, prompt):
    # Mock responses for different models
    mock_responses = {
        "LLaMA": "The future of AI is promising, with advancements in reasoning and efficiency.",
        "Mistral": "AI will revolutionize industries, enhancing automation and decision-making.",
        "BERT": "AI's future lies in better understanding context and human-like interactions."
    }
    
    # Simulate processing time (randomized for demonstration)
    processing_time = random.uniform(0.5, 2.0)
    time.sleep(processing_time)  # Simulate model inference delay
    
    # Get the response for the given model
    response = mock_responses.get(model_name, "No response available.")
    
    return response, processing_time

# Analyze response metrics
def analyze_response(response):
    # Calculate response length (in words)
    word_count = len(response.split())
    
    # Perform basic sentiment analysis
    sentiment = TextBlob(response).sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)
    
    return word_count, sentiment

# Main function to compare models
def compare_models(prompt):
    models = ["LLaMA", "Mistral", "BERT"]
    results = []
    
    print(f"\nPrompt: {prompt}\n")
    
    for model in models:
        # Get response and processing time
        start_time = time.time()
        response, processing_time = simulate_model_response(model, prompt)
        end_time = time.time()
        
        # Analyze response
        word_count, sentiment = analyze_response(response)
        
        # Store results
        results.append({
            "Model": model,
            "Response": response,
            "Word Count": word_count,
            "Sentiment": round(sentiment, 2),
            "Processing Time (s)": round(processing_time, 2)
        })
        
        # Print individual model result
        print(f"Model: {model}")
        print(f"Response: {response}")
        print(f"Word Count: {word_count}")
        print(f"Sentiment Score: {round(sentiment, 2)}")
        print(f"Processing Time: {round(processing_time, 2)} seconds\n")
    
    # Create comparison table
    headers = ["Model", "Response", "Word Count", "Sentiment", "Processing Time (s)"]
    table_data = [[r["Model"], r["Response"], r["Word Count"], r["Sentiment"], r["Processing Time (s)"]] for r in results]
    
    print("Comparison Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Run the comparison
if __name__ == "__main__":
    try:
        compare_models(prompt)
    except Exception as e:
        print(f"An error occurred: {e}")

# Requirements: Install dependencies
# pip install textblob tabulate