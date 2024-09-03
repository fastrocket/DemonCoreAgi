import gradio as gr
import json
import sqlite3
import time
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import threading

load_dotenv()

# Thread-local storage
local = threading.local()

# Initialize database
def get_db():
    if not hasattr(local, "db"):
        local.db = sqlite3.connect('results.db', check_same_thread=False)
    return local.db

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS results
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      result_number INTEGER,
                      raw_result TEXT,
                      analysis TEXT)''')
        conn.commit()

init_db()

# Load the master prompt
with open('master_prompt.txt', 'r') as file:
    MASTER_PROMPT = file.read()

def query_llm(prompt, model="openai"):
    if model == "openai":
        chat = ChatOpenAI(model_name="gpt-4o-mini", streaming=True)
    elif model == "anthropic":
        chat = ChatAnthropic(model="claude-2")
    else:
        raise ValueError("Invalid model specified")

    messages = [
        SystemMessage(content=MASTER_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = chat.invoke(messages)
    return response.content

def analyze_result(result):
    prompt = f"Analyze the following result and explain briefly if the AI followed the instructions and produced correct JSON:\n\n{result}"
    return query_llm(prompt)

def save_result(result, analysis):
    with get_db() as conn:
        c = conn.cursor()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        c.execute("SELECT MAX(result_number) FROM results")
        max_number = c.fetchone()[0]
        result_number = 1 if max_number is None else max_number + 1
        
        c.execute("INSERT INTO results (timestamp, result_number, raw_result, analysis) VALUES (?, ?, ?, ?)",
                  (timestamp, result_number, result, analysis))
        conn.commit()

def get_results():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, timestamp, result_number FROM results ORDER BY id DESC")
        return [{"id": row[0], "timestamp": row[1], "result_number": row[2]} for row in c.fetchall()]

def load_result(result_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT raw_result, analysis FROM results WHERE id = ?", (result_id,))
        row = c.fetchone()
        return {"raw_result": row[0], "analysis": row[1]} if row else None

def interface(request, feedback, model_choice, result_id):
    if request:
        result = query_llm(request, model_choice)
        analysis = analyze_result(result)
        save_result(result, analysis)
        
        # Update the results list
        results_list = get_results()
        
        return result, analysis, results_list, None
    elif result_id:
        result = load_result(result_id)
        if result:
            return result["raw_result"], result["analysis"], gr.update(), gr.update()
        else:
            return "", "", gr.update(), gr.update()
    else:
        return "", "", gr.update(), gr.update()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AutonomousGPT Interface")
    
    with gr.Row():
        with gr.Column(scale=2):
            request_input = gr.Textbox(label="Enter your request")
            feedback_input = gr.Textbox(label="Feedback")
            model_choice = gr.Radio(["openai", "anthropic"], label="Choose LLM", value="openai")
            submit_btn = gr.Button("Next Step")
        
        with gr.Column(scale=1):
            results_dropdown = gr.Dropdown(choices=get_results(), label="Select a result", interactive=True)
    
    with gr.Row():
        with gr.Column(scale=2):
            result_output = gr.Textbox(label="Result")
        with gr.Column(scale=1):
            analysis_output = gr.Textbox(label="Analysis")
    
    submit_btn.click(interface, 
                     inputs=[request_input, feedback_input, model_choice, results_dropdown], 
                     outputs=[result_output, analysis_output, results_dropdown, results_dropdown])
    
    results_dropdown.change(interface,
                            inputs=[request_input, feedback_input, model_choice, results_dropdown],
                            outputs=[result_output, analysis_output, results_dropdown, results_dropdown])

if __name__ == "__main__":
    demo.launch()
