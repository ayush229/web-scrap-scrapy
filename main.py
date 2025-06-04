from flask import Flask, request, jsonify, make_response
from functools import wraps
import logging
import os
from together import Together
from urllib.parse import urlparse, urljoin
import uuid
import re
import traceback
import json
from flask_cors import CORS
import subprocess
import sys

app = Flask(__name__)
CORS(app, origins=["https://agent-ai-production-b4d6.up.railway.app","https://agent-ai-production-b4d6.up.railway.app/agent"], supports_credentials=True, methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"], allow_headers=["Authorization", "Content-Type"])

# --- Configuration ---
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "ayush1")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "blackbox098")
SCRAPED_DATA_DIR = "scraped_content"
# Ensure the directory exists
os.makedirs(SCRAPED_DATA_DIR, exist_ok=True)

# --- Initialize Clients and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    client = Together()
except Exception as e:
    logger.critical(f"FATAL: Could not initialize Together client. Ensure TOGETHER_API_KEY environment variable is set. Error: {e}")
    client = None

# --- Authentication Decorator ---
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not (auth.username == AUTH_USERNAME and auth.password == AUTH_PASSWORD):
            logger.warning("Authentication failed.")
            return make_response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

# --- Helper Functions ---

def ensure_url_scheme(url):
    """
    Ensures that a URL has a scheme (e.g., http:// or https://).
    Defaults to https:// if no scheme is present.
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        # Default to https if no scheme is provided
        return f"https://{url}"
    return url

def get_stored_content(unique_code):
    """Retrieves the full content of a stored agent file."""
    file_path = os.path.join(SCRAPED_DATA_DIR, f"{unique_code}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    return None

def ask_llama(prompt, model="meta-llama/Llama-3-8b-chat-hf"):
    """Sends a prompt to the Together AI LLM."""
    if not client:
        logger.error("Together AI client not initialized. Cannot ask Llama.")
        return "Error: AI client not available."
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the provided web content."},
                {"role": "user", "content": prompt},
            ],
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error asking Llama: {e}")
        return f"Error communicating with AI: {e}"

def find_relevant_content(scraped_data, query):
    """
    Finds relevant sections in scraped data based on query keywords.
    Improved to handle the new Scrapy output structure.
    """
    if not scraped_data or "results" not in scraped_data:
        return []

    relevant_sections = []
    query_words = set(re.findall(r'\b\w+\b', query.lower()))

    # Iterate through each page's results
    for page_result in scraped_data.get("results", []):
        if page_result.get("type") == "beautify" and "data" in page_result:
            for section in page_result["data"].get("sections", []):
                section_text = []
                if section.get("heading") and section["heading"].get("text"):
                    section_text.append(section["heading"]["text"].lower())
                section_text.extend([p.lower() for p in section.get("content", [])])

                if any(word in ' '.join(section_text) for word in query_words):
                    relevant_sections.append({
                        "url": page_result["url"],
                        "heading": section.get("heading"),
                        "content": section.get("content")
                    })
    return relevant_sections

def find_relevant_sentences(text_blocks, query):
    """A placeholder for more sophisticated sentence relevance."""
    relevant_sents = []
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    for block in text_blocks:
        sentences = re.split(r'(?<=[.!?])\s+', block)
        for sent in sentences:
            if any(word in sent.lower() for word in query_words):
                relevant_sents.append(sent)
    return relevant_sents

def run_scrapy_spider(urls_str, scrape_mode, crawl_enabled, max_pages, unique_code):
    """
    Executes the Scrapy spider as a subprocess and reads its JSON output.
    Returns a dictionary with status and results or None on error.
    """
    output_filepath = os.path.join(SCRAPED_DATA_DIR, f"{unique_code}.json")

    command = [
        sys.executable,
        '-m', 'scrapy.cmdline', 'runspider',
        'scrapy_spider.py',
        '-a', f'start_urls={urls_str}',
        '-a', f'scrape_mode={scrape_mode}',
        '-a', f'crawl_enabled={str(crawl_enabled).lower()}',
        '-a', f'max_pages={max_pages}',
        '-a', f'output_file={output_filepath}'
    ]

    logger.info(f"Running Scrapy command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600
        )
        logger.info(f"Scrapy process finished. Stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Scrapy process stderr: {result.stderr}")

        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
                # Ensure a consistent return structure
                return {"status": "success", "results": scraped_data}
        else:
            logger.error(f"Scrapy output file not found after execution: {output_filepath}")
            return {"status": "error", "error": "Scrapy output file not found."}

    except subprocess.CalledProcessError as e:
        logger.error(f"Scrapy command failed with exit code {e.returncode}: {e.stderr}")
        return {"status": "error", "error": f"Scrapy command failed: {e.stderr}"}
    except subprocess.TimeoutExpired:
        logger.error(f"Scrapy command timed out after 600 seconds for URLs: {urls_str}")
        return {"status": "error", "error": "Scrapy command timed out."}
    except FileNotFoundError:
        logger.error("Scrapy command or Python executable not found. Make sure Scrapy is installed and in PATH.")
        return {"status": "error", "error": "Scrapy or Python executable not found."}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from Scrapy output: {e} for file {output_filepath}")
        return {"status": "error", "error": f"Error decoding Scrapy output: {e}"}
    except Exception as e:
        logger.error(f"An unexpected error occurred while running Scrapy: {e}\n{traceback.format_exc()}")
        return {"status": "error", "error": f"An unexpected error occurred during scraping: {e}"}

# --- API Endpoints ---

@app.route('/scrape_and_store', methods=['POST'])
@requires_auth
def scrape_and_store():
    data = request.get_json()
    urls_input_str = data.get('urls')
    agent_name = data.get('agent_name')
    existing_unique_code = data.get('unique_code')

    if not urls_input_str or not agent_name:
        logger.warning("Missing URLs or agent_name for /scrape_and_store")
        return jsonify({"status": "error", "error": "Missing 'urls' or 'agent_name'"}), 400

    urls_list_with_scheme = [ensure_url_scheme(u.strip()) for u in urls_input_str.split(',') if u.strip()]
    urls_str_for_spider = ",".join(urls_list_with_scheme)

    unique_code = existing_unique_code if existing_unique_code else str(uuid.uuid4())
    output_filename = f"{unique_code}.json"
    output_filepath = os.path.join(SCRAPED_DATA_DIR, output_filename)

    logger.info(f"Initiating scrape_and_store for agent '{agent_name}' ({unique_code}) on URLs: {urls_str_for_spider}")

    scrapy_result = run_scrapy_spider(
        urls_str=urls_str_for_spider,
        scrape_mode='beautify',
        crawl_enabled=True,
        max_pages=50,
        unique_code=unique_code
    )

    if scrapy_result and scrapy_result.get("status") == "success":
        agent_metadata = {
            "agent_name": agent_name,
            "urls_scraped": urls_str_for_spider,
            "timestamp": scrapy_result.get("closure_reason", "unknown") # Use closure_reason or a proper timestamp
        }

        final_data_to_store = {
            "agent_metadata": agent_metadata,
            "scraped_data": scrapy_result # Store the full Scrapy output structure including status
        }

        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(final_data_to_store, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully stored agent '{agent_name}' data to {output_filepath}")
            return jsonify({
                "status": "success",
                "message": f"Successfully scraped and stored content for agent '{agent_name}'",
                "unique_code": unique_code,
                "url": urls_str_for_spider,
                "data": final_data_to_store
            })
        except Exception as e:
            error_message = f"Error saving scraped content for {unique_code}: {e}"
            logger.error(error_message)
            return jsonify({"status": "error", "error": error_message}), 500
    else:
        error_message = scrapy_result.get("error", "Unknown error during Scrapy execution.") if scrapy_result else "Scrapy did not return a valid result."
        logger.error(f"Scrapy execution failed for /scrape_and_store: {error_message}")
        return jsonify({"status": "error", "error": error_message}), 500


@app.route('/ask_stored', methods=['POST'])
@requires_auth
def ask_stored():
    data = request.get_json()
    unique_code = data.get('unique_code')
    user_query = data.get('user_query')

    if not unique_code or not user_query:
        logger.warning("Missing unique_code or user_query for /ask_stored")
        return jsonify({"status": "error", "error": "Missing 'unique_code' or 'user_query'"}), 400

    logger.info(f"Request to ask_stored for code: {unique_code} with query: {user_query}")

    stored_data = get_stored_content(unique_code)
    if not stored_data:
        logger.warning(f"No stored content found for code: {unique_code}")
        return jsonify({"status": "error", "error": f"No content found for unique_code: {unique_code}"}), 404

    scraped_data_from_file = stored_data.get("scraped_data", {})
    if not scraped_data_from_file or "results" not in scraped_data_from_file:
        logger.warning(f"No scraped data found within stored file for code: {unique_code}")
        return jsonify({"status": "error", "error": f"No scraped data found within stored file for unique_code: {unique_code}"}), 404

    relevant_sections = find_relevant_content(scraped_data_from_file, user_query)

    if not relevant_sections:
        response_text = "No relevant content found for your query in the stored data."
        logger.info("No relevant sections found, returning generic response.")
    else:
        context = []
        for section in relevant_sections[:5]:
            context.append(f"URL: {section['url']}")
            if section['heading'] and section['heading'].get('text'):
                context.append(f"Heading: {section['heading']['text']}")
            context.extend(section.get('content', []))
            context.append("\n---\n")

        prompt = f"Based on the following web content, answer the question accurately and concisely. If the information is not directly available, state that. \n\nContext:\n{''.join(context)}\n\nQuestion: {user_query}"

        logger.info(f"Sending prompt to LLM for code: {unique_code}")
        response_text = ask_llama(prompt)

    return jsonify({"status": "success", "answer": response_text, "relevant_sections": relevant_sections})


@app.route('/scrape', methods=['GET', 'POST'])
@requires_auth
def scrape():
    if request.method == 'POST':
        data = request.get_json()
        urls_input_str = data.get('url')
        req_type = data.get('type', 'beautify')
        max_pages = int(data.get('max_pages', 1))
    else:
        urls_input_str = request.args.get('url')
        req_type = request.args.get('type', 'beautify')
        max_pages = int(request.args.get('max_pages', 1))

    if not urls_input_str:
        logger.warning("Missing URL for /scrape")
        return jsonify({"status": "error", "error": "Missing 'url' parameter"}), 400

    urls_list_with_scheme = [ensure_url_scheme(u.strip()) for u in urls_input_str.split(',') if u.strip()]
    urls_str_for_spider = ",".join(urls_list_with_scheme)

    if not urls_str_for_spider:
        logger.warning("Invalid URLs provided for /scrape after scheme check")
        return jsonify({"status": "error", "error": "Invalid URLs provided"}), 400

    temp_unique_code = str(uuid.uuid4())

    crawl_enabled = False
    scrape_mode = 'beautify'
    ai_enabled = False

    if req_type == 'raw':
        scrape_mode = 'raw'
    elif req_type == 'beautify':
        scrape_mode = 'beautify'
    elif req_type == 'crawl_raw':
        scrape_mode = 'raw'
        crawl_enabled = True
    elif req_type == 'crawl_beautify':
        scrape_mode = 'beautify'
        crawl_enabled = True
    elif req_type == 'ai':
        scrape_mode = 'beautify'
        ai_enabled = True
    elif req_type == 'crawl_ai':
        scrape_mode = 'beautify'
        crawl_enabled = True
        ai_enabled = True
    else:
        logger.warning(f"Invalid type: {req_type} for /scrape")
        return jsonify({"status": "error", "error": "Invalid 'type' parameter. Valid types are 'raw', 'beautify', 'crawl_raw', 'crawl_beautify', 'ai', 'crawl_ai'"}), 400

    if not crawl_enabled:
        max_pages = 1

    scrapy_output = run_scrapy_spider(
        urls_str=urls_str_for_spider,
        scrape_mode=scrape_mode,
        crawl_enabled=crawl_enabled,
        max_pages=max_pages,
        unique_code=temp_unique_code
    )

    temp_filepath = os.path.join(SCRAPED_DATA_DIR, f"{temp_unique_code}.json")
    if os.path.exists(temp_filepath):
        try:
            os.remove(temp_filepath)
            logger.info(f"Cleaned up temporary Scrapy output file: {temp_filepath}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {temp_filepath}: {e}")

    if scrapy_output and scrapy_output.get("status") == "success":
        extracted_data = scrapy_output.get("results", [])

        if ai_enabled:
            full_text_content = []
            for page_res in extracted_data:
                if page_res.get("type") == "beautify" and "data" in page_res:
                    for section in page_res["data"].get("sections", []):
                        if section.get("heading") and section["heading"].get("text"):
                            full_text_content.append(section["heading"]["text"])
                        full_text_content.extend(section.get("content", []))

            combined_text = "\n\n".join(full_text_content)

            if not combined_text.strip():
                ai_answer = "No readable content found on the page(s) to process with AI."
            else:
                user_query = data.get('user_query', "Summarize the content.")
                if not user_query:
                    user_query = "Summarize the content."

                ai_prompt = f"Based on the following web content, provide a concise answer or summary for: '{user_query}'. If the information is not directly available, state that. \n\nContent:\n{combined_text}"
                ai_answer = ask_llama(ai_prompt)

            return jsonify({
                "status": "success",
                "url": urls_str_for_spider,
                "type": req_type,
                "ai_response": ai_answer,
                "scraped_data_summary": [{"url": d["url"], "type": d["type"]} for d in extracted_data]
            })
        else:
            return jsonify({
                "status": "success",
                "url": urls_str_for_spider,
                "type": req_type,
                "data": extracted_data
            })
    else:
        error_message = scrapy_output.get("error", "Unknown error during Scrapy execution.") if scrapy_output else "Scrapy did not return a valid result."
        logger.error(f"Scrapy execution failed for /scrape: {error_message}")
        return jsonify({"status": "error", "error": error_message}), 500

@app.route('/agents', methods=['GET'])
@requires_auth
def list_agents():
    """Lists all stored agents (unique codes and names)."""
    agents = []
    for filename in os.listdir(SCRAPED_DATA_DIR):
        if filename.endswith(".json"):
            unique_code = filename[:-5]
            file_path = os.path.join(SCRAPED_DATA_DIR, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    agent_metadata = data.get("agent_metadata", {})
                    agents.append({
                        "unique_code": unique_code,
                        "agent_name": agent_metadata.get("agent_name", "Unnamed Agent"),
                        "urls_scraped": agent_metadata.get("urls_scraped", ""),
                        "timestamp": agent_metadata.get("timestamp", "N/A")
                    })
            except json.JSONDecodeError as e:
                logger.error(f"Skipping malformed JSON file {filename}: {e}")
            except Exception as e:
                logger.error(f"Error reading agent file {filename}: {e}")
    return jsonify({"status": "success", "agents": agents})

@app.route('/agents/<unique_code>', methods=['PUT'])
@requires_auth
def update_agent(unique_code):
    data = request.get_json()
    new_urls_input_str = data.get('urls')
    new_agent_name = data.get('agent_name')

    if not new_urls_input_str:
        return jsonify({"status": "error", "error": "Missing 'urls' for update"}), 400

    new_urls_list_with_scheme = [ensure_url_scheme(u.strip()) for u in new_urls_input_str.split(',') if u.strip()]
    new_urls_str_for_spider = ",".join(new_urls_list_with_scheme)

    logger.info(f"Updating agent {unique_code} with new URLs: {new_urls_str_for_spider}")

    existing_file_path = os.path.join(SCRAPED_DATA_DIR, f"{unique_code}.json")
    if not os.path.exists(existing_file_path):
        return jsonify({"status": "error", "error": f"Agent with unique_code {unique_code} not found for update"}), 404

    scrapy_result = run_scrapy_spider(
        urls_str=new_urls_str_for_spider,
        scrape_mode='beautify',
        crawl_enabled=True,
        max_pages=50,
        unique_code=unique_code
    )

    if scrapy_result and scrapy_result.get("status") == "success":
        agent_metadata = {
            "agent_name": new_agent_name if new_agent_name else "Unnamed Agent",
            "urls_scraped": new_urls_str_for_spider,
            "timestamp": scrapy_result.get("closure_reason", "updated")
        }

        final_data_to_store = {
            "agent_metadata": agent_metadata,
            "scraped_data": scrapy_result
        }

        try:
            with open(existing_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_data_to_store, f, ensure_ascii=False, indent=4)
            logger.info(f"Successfully updated agent {unique_code} data.")
            return jsonify({"status": "success", "message": f"Agent {unique_code} updated successfully."})
        except Exception as e:
            error_message = f"Error saving updated content for {unique_code}: {e}"
            logger.error(error_message)
            return jsonify({"status": "error", "error": error_message}), 500
    else:
        error_message = scrapy_result.get("error", "Unknown error during Scrapy execution.") if scrapy_result else "Scrapy did not return a valid result."
        logger.error(f"Scrapy execution failed during update for {unique_code}: {error_message}")
        return jsonify({"status": "error", "error": error_message}), 500


@app.route('/agents/<unique_code>', methods=['DELETE'])
@requires_auth
def delete_agent(unique_code):
    """Deletes an agent's stored data file."""
    logger.info(f"Request to delete agent with code: {unique_code}")
    file_path = os.path.join(SCRAPED_DATA_DIR, f"{unique_code}.json")
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Successfully deleted agent file: {file_path}")
            return jsonify({"status": "success", "message": f"Agent {unique_code} and its data deleted."})
        else:
            logger.warning(f"Attempted to delete non-existent agent file: {unique_code}")
            return jsonify({"status": "error", "error": f"Agent with unique_code {unique_code} not found."}), 404
    except OSError as e:
        error_message = f"Error deleting agent data file {unique_code}: {e}"
        logger.error(error_message)
        return jsonify({"status": "error", "error": "Could not delete agent data file"}), 500
    except Exception as e:
         error_message = f"Unexpected error during agent deletion {unique_code}: {e}\n{traceback.format_exc()}"
         logger.error(error_message)
         print(error_message)
         return jsonify({"status": "error", "error": "An unexpected error occurred during deletion"}), 500


@app.route('/get_stored_file/<unique_code>', methods=['GET'])
@requires_auth
def get_stored_file(unique_code):
    """Retrieves the full content of a stored agent file."""
    logger.info(f"Request to get stored file for code: {unique_code}")
    content = get_stored_content(unique_code)
    if content:
        return jsonify({"status": "success", "unique_code": unique_code, "content": content})
    else:
        logger.warning(f"Stored file not found for code: {unique_code}")
        return jsonify({"status": "error", "error": f"Content not found for unique_code: {unique_code}"}), 404


# --- Main Execution ---
if __name__ == '__main__':
    print(f"Starting Flask server on host 0.0.0.0 port 5000")
    print(f"Serving scraped data from: {os.path.abspath(SCRAPED_DATA_DIR)}")
    app.run(host='0.0.0.0', port=5000, debug=False)
