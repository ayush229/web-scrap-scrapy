import os
import re
import json
import uuid
import sys
import traceback
import subprocess
import logging
from functools import wraps
from urllib.parse import urlparse, urljoin

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://agent-ai-production-b4d6.up.railway.app","https://agent-ai-production-b4d6.up.railway.app/agent"], supports_credentials=True, methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"], allow_headers=["Authorization", "Content-Type"])

# --- Configuration ---
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "ayush1")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "blackbox098")
SCRAPED_DATA_DIR = "scraped_content"
os.makedirs(SCRAPED_DATA_DIR, exist_ok=True)

# --- Initialize Clients and Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to initialize Together AI client
try:
    from together import Together
    client = Together()
except ImportError:
    logger.critical("Together AI client library 'together' not found. Please install it with 'pip install together'.")
    client = None
except Exception as e:
    logger.critical(f"FATAL: Could not initialize Together client. Ensure TOGETHER_API_KEY environment variable is set. Error: {e}")
    client = None

# --- Authentication Decorator ---
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Allow OPTIONS requests to pass without authentication
        if request.method == 'OPTIONS':
            return '', 200

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

def ask_llama(prompt, model="meta-llama/Llama-3-8b-chat-hf"): # Keeping Llama-3-8b-chat-hf as it was in file2
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

def find_relevant_content(scraped_data_results, query):
    """
    Finds relevant sections in scraped data based on query keywords.
    Improved to handle the new Scrapy output structure.
    """
    if not scraped_data_results:
        return [], False

    relevant_sections = []
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    meaningful_match_found = False

    # Define stop words to identify meaningful query words
    stop_words = set([
        "a", "an", "the", "and", "or", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
        "shall", "should", "may", "might", "must", "it's", "don't", "i'm", "you're",
        "he's", "she's", "we're", "they're", "isn't", "aren't", "wasn't", "weren't",
        "haven't", "hasn't", "hadn't", "don't", "doesn't", "didn't", "can't", "couldn't",
        "won't", "wouldn't", "shan't", "shouldn't", "mayn't", "mightn't", "mustn't",
        "you", "i", "he", "she", "it", "we", "they", "this", "that", "these", "those",
        "my", "your", "his", "her", "its", "our", "their", "here", "there", "what",
        "where", "when", "why", "how", "who", "whom", "whose", "with", "without",
        "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "many", "more", "most", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "m", "d", "ll", "re", "ve", "y",
    ])
    meaningful_query_tokens = {token for token in query_words if token not in stop_words}

    if not meaningful_query_tokens: # If query contains only stop words
        return [], False

    for page_result in scraped_data_results:
        if page_result.get("type") == "beautify" and "data" in page_result:
            page_is_relevant = False
            page_text = ""
            for section in page_result["data"].get("sections", []):
                if section.get("heading") and section["heading"].get("text"):
                    page_text += " " + section["heading"]["text"].lower()
                page_text += " " + ' '.join([p.lower() for p in section.get("content", [])])

            for token in meaningful_query_tokens:
                if re.search(r'\b' + re.escape(token) + r'\b', page_text):
                    page_is_relevant = True
                    meaningful_match_found = True
                    break

            if page_is_relevant:
                # Reconstruct content structure for consistency with file1
                content_for_page = []
                for section in page_result["data"].get("sections", []):
                    heading_data = section.get("heading")
                    heading_text = heading_data.get("text", "") if isinstance(heading_data, dict) else heading_data
                    content_for_page.append({
                        "heading": heading_text or None,
                        "paragraphs": section.get("content", [])
                    })
                relevant_sections.append({"url": page_result["url"], "content": content_for_page})
        elif page_result.get("type") == "raw" and "data" in page_result:
            page_is_relevant = False
            raw_text = page_result["data"].lower()
            for token in meaningful_query_tokens:
                if re.search(r'\b' + re.escape(token) + r'\b', raw_text):
                    page_is_relevant = True
                    meaningful_match_found = True
                    break
            if page_is_relevant:
                relevant_sections.append({"url": page_result["url"], "raw_data": page_result["data"]})


    return relevant_sections, meaningful_match_found


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
        '-o', output_filepath # Use -o for direct output to file
    ]

    logger.info(f"Running Scrapy command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=600 # 10 minutes timeout
        )
        logger.info(f"Scrapy process finished. Stdout: {result.stdout}")
        if result.stderr:
            logger.error(f"Scrapy process stderr: {result.stderr}")

        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
                return {"status": "success", "results": scraped_data} # Scrapy results are already a list of dictionaries
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
    """
    Scrapes content from provided URLs using Scrapy, associates it with an agent name,
    stores it, and returns a unique code and agent name.
    """
    try:
        data = request.get_json(force=True) or {}
        urls_input_str = data.get('url') # Match file1's parameter name
        agent_name = data.get('agent_name')

        if not urls_input_str:
            return jsonify({"status": "error", "error": "URL parameter is required"}), 400
        if not agent_name:
            return jsonify({"status": "error", "error": "agent_name parameter is required"}), 400

        urls_list_with_scheme = [ensure_url_scheme(u.strip()) for u in urls_input_str.split(',') if u.strip()]
        if not urls_list_with_scheme:
            return jsonify({"status": "error", "error": "No valid URLs provided"}), 400

        urls_str_for_spider = ",".join(urls_list_with_scheme)
        unique_code = str(uuid.uuid4())
        output_filepath = os.path.join(SCRAPED_DATA_DIR, f"{unique_code}.json") # Store as .json

        logger.info(f"Initiating scrape_and_store for agent '{agent_name}' ({unique_code}) on URLs: {urls_str_for_spider}")

        scrapy_result = run_scrapy_spider(
            urls_str=urls_str_for_spider,
            scrape_mode='beautify', # Assuming beautify is the default for scrape_and_store
            crawl_enabled=False, # scrape_and_store typically does not crawl beyond initial URLs
            max_pages=1, # Only scrape the provided URLs
            unique_code=unique_code
        )

        if scrapy_result and scrapy_result.get("status") == "success":
            # The 'results' key from run_scrapy_spider contains the list of scraped items
            scraped_items = scrapy_result.get("results", [])

            # Transform Scrapy output to match file1's structure for 'results'
            transformed_results = []
            scrape_errors = []
            for item in scraped_items:
                url = item.get("url", "Unknown URL")
                item_type = item.get("type")
                item_data = item.get("data")

                if item_type == "beautify" and isinstance(item_data, dict) and "sections" in item_data:
                    page_content = []
                    for section in item_data.get("sections", []):
                        heading_data = section.get("heading")
                        heading_text = heading_data.get("text", "") if isinstance(heading_data, dict) else heading_data
                        page_content.append({
                            "heading": heading_text or None,
                            "paragraphs": section.get("content", [])
                        })
                    transformed_results.append({"url": url, "content": page_content})
                elif item_type == "raw" and isinstance(item_data, str):
                    transformed_results.append({"url": url, "raw_data": item_data})
                else:
                    logger.warning(f"Unexpected Scrapy item format for URL {url}: {item}")
                    scrape_errors.append({"url": url, "error": "Unexpected Scrapy item format"})
            
            # This is the data structure for the stored JSON file
            data_to_store = {
                "agent_name": agent_name,
                "urls": urls_list_with_scheme, # Store the list of URLs attempted
                "results": transformed_results,
                "errors": scrape_errors
            }

            try:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    json.dump(data_to_store, f, ensure_ascii=False, indent=4)
                logger.info(f"Successfully stored agent '{agent_name}' data to {output_filepath}")
                
                return jsonify({
                    "status": "success",
                    "unique_code": unique_code,
                    "agent_name": agent_name,
                    "scrape_errors": scrape_errors # Inform user about any URLs that failed
                }), 201
            except Exception as e:
                error_message = f"Error saving scraped content for {unique_code}: {e}"
                logger.error(error_message)
                return jsonify({"status": "error", "error": "Failed to store scraped data"}), 500
        else:
            error_message = scrapy_result.get("error", "Unknown error during Scrapy execution.") if scrapy_result else "Scrapy did not return a valid result."
            logger.error(f"Scrapy execution failed for /scrape_and_store: {error_message}")
            return jsonify({"status": "error", "error": error_message}), 500

    except Exception as e:
        error_message = f"Internal server error in /scrape_and_store: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"status": "error", "error": "An internal server error occurred"}), 500


@app.route('/ask_stored', methods=['POST'])
@requires_auth
def ask_stored():
    """
    Answers a user query based on previously stored scraped content identified by unique_code.
    """
    try:
        data = request.get_json(force=True) or {}
        unique_code = data.get('unique_code')
        user_query = data.get('user_query')

        if not unique_code:
            return jsonify({"status": "error", "error": "unique_code parameter is required"}), 400
        if not user_query:
            return jsonify({"status": "error", "error": "user_query parameter is required"}), 400

        logger.info(f"ask_stored request for code '{unique_code}' with query: '{user_query}'")

        stored_data = get_stored_content(unique_code)
        if not stored_data:
            logger.warning(f"Content not found for unique_code: {unique_code}")
            return jsonify({"status": "error", "error": f"Content not found for unique_code: {unique_code}"}), 404

        scraped_results = stored_data.get('results', []) # Get the 'results' list from the stored data
        if not scraped_results:
            logger.warning(f"No scrape results found in stored data for unique_code: {unique_code}")
            return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response (no content available).", "ai_used": False})

        # Use the find_relevant_content which is now adapted to handle Scrapy's output format
        relevant_content_objects, meaningful_match_found = find_relevant_content(scraped_results, user_query)

        if not relevant_content_objects or not meaningful_match_found:
            logger.info(f"No relevant content found or only stop words matched for query '{user_query}' in {unique_code}.")
            return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response based on the stored content and your query.", "ai_used": False})

        logger.info(f"Found {len(relevant_content_objects)} relevant content objects for query.")

        prompt_text = f"""As a knowledgeable agent, please provide a direct and conversational answer to the user's question based *only* on the provided website content below.
Do not mention that you are using the provided information.
If the answer is not found in the text, state that you cannot provide a helpful response based on the available information.
User question: "{user_query}"
Website content: """
        content_added = False
        for i, content_obj in enumerate(relevant_content_objects):
            prompt_text += f"\n--- Content from {content_obj.get('url', 'Unknown URL')} ---\n"
            if content_obj.get('content'): # This means it's a beautify result
                for section in content_obj.get('content', []):
                    heading = section.get('heading', '') or ""
                    paragraphs = section.get('paragraphs', []) or []
                    if heading:
                        prompt_text += f"Heading: {heading}\n"
                        content_added = True
                    if paragraphs:
                        prompt_text += "\n".join(paragraphs) + "\n"
                        content_added = True
            elif content_obj.get('raw_data'): # This means it's a raw result
                prompt_text += content_obj['raw_data'] + "\n"
                content_added = True
            prompt_text += "--- End of Content ---\n"

        if not content_added:
            logger.warning(f"Relevant content objects found, but no actual text could be extracted for the prompt (code: {unique_code}).")
            return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response due to an issue processing the stored content.", "ai_used": False})

        ai_response = ask_llama(prompt_text)

        unhelpful_phrases = ["sorry, i am unable", "cannot provide a helpful response", "no information available", "based on the text provided", "information is not available"]
        is_unhelpful = not ai_response or len(ai_response.strip()) < 15 or any(phrase in ai_response.lower() for phrase in unhelpful_phrases)

        if is_unhelpful:
            logger.info(f"LLM response deemed unhelpful for code {unique_code}, query '{user_query}'. Response: '{ai_response}'")
            return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response based on the available information.", "ai_used": True})
        else:
            logger.info(f"LLM provided a response for code {unique_code}, query '{user_query}'.")
            return jsonify({"status": "success", "ai_response": ai_response, "ai_used": True})

    except Exception as e:
        error_message = f"Internal server error in /ask_stored: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"status": "error", "error": "An internal server error occurred"}), 500


@app.route('/scrape', methods=['GET', 'POST'])
@requires_auth
def scrape():
    """
    General purpose endpoint for scraping or crawling URLs using Scrapy.
    Supports types: 'raw', 'beautify', 'ai', 'crawl_raw', 'crawl_beautify', 'crawl_ai'.
    'ai' types require a 'user_query'.
    """
    try:
        if request.method == 'GET':
            urls_str = request.args.get('url')
            content_type = request.args.get('type', 'beautify').lower()
            user_query = request.args.get('user_query', '')
        else: # POST
            try:
                data = request.get_json(force=True) or {}
            except Exception as e:
                error_message = f"Error parsing JSON in POST /scrape: {e}"
                logger.error(error_message)
                return jsonify({"status": "error", "error": "Invalid JSON payload"}), 400
            urls_str = data.get('url', '')
            content_type = data.get('type', 'beautify').lower()
            user_query = data.get('user_query', '')

        if not urls_str:
            return jsonify({"status": "error", "error": "URL parameter is required"}), 400

        urls = [ensure_url_scheme(u.strip()) for u in urls_str.split(',') if u.strip()]
        if not urls:
            return jsonify({"status": "error", "error": "No valid URLs provided"}), 400

        logger.info(f"/scrape request - Type: {content_type}, URLs: {urls}, Query: '{user_query if user_query else 'N/A'}'")

        urls_str_for_spider = ",".join(urls)
        unique_code = str(uuid.uuid4()) # Generate a new unique code for temporary scrape results

        # --- Handle Simple Scrape Types ('raw', 'beautify') ---
        if content_type in ['raw', 'beautify']:
            scrapy_result = run_scrapy_spider(
                urls_str=urls_str_for_spider,
                scrape_mode=content_type,
                crawl_enabled=False,
                max_pages=1, # Only scrape the provided URLs
                unique_code=unique_code
            )

            if scrapy_result and scrapy_result.get("status") == "success":
                # Scrapy results are already a list of dictionaries, format for response
                formatted_results = []
                for item in scrapy_result.get("results", []):
                    if item.get("type") == "beautify" and "data" in item:
                        # Reconstruct content similar to file1's scrape_website output
                        page_content = []
                        for section in item["data"].get("sections", []):
                            heading_data = section.get("heading")
                            heading_text = heading_data.get("text", "") if isinstance(heading_data, dict) else heading_data
                            page_content.append({
                                "heading": heading_text or None,
                                "paragraphs": section.get("content", [])
                            })
                        formatted_results.append({"url": item.get("url"), "content": page_content})
                    elif item.get("type") == "raw" and "data" in item:
                        formatted_results.append({"url": item.get("url"), "raw_data": item.get("data")})
                
                return jsonify({"status": "success", "results": formatted_results})
            else:
                error_message = scrapy_result.get("error", "Unknown error during Scrapy execution.") if scrapy_result else "Scrapy did not return a valid result."
                logger.error(f"Scrapy execution failed for /scrape (type: {content_type}): {error_message}")
                return jsonify({"status": "error", "error": error_message}), 500

        # --- Handle AI Scrape Type ('ai') ---
        elif content_type == 'ai':
            if not user_query:
                return jsonify({"status": "error", "error": "user_query is required for 'ai' type"}), 400

            scrapy_result = run_scrapy_spider(
                urls_str=urls_str_for_spider,
                scrape_mode='beautify', # Typically 'beautify' for AI content
                crawl_enabled=False,
                max_pages=1,
                unique_code=unique_code
            )

            if scrapy_result and scrapy_result.get("status") == "success":
                scraped_items = scrapy_result.get("results", [])
                
                # Transform Scrapy output to match file1's structure for find_relevant_content
                transformed_scraped_data_for_ai = []
                for item in scraped_items:
                    url = item.get("url", "Unknown URL")
                    item_type = item.get("type")
                    item_data = item.get("data")

                    if item_type == "beautify" and isinstance(item_data, dict) and "sections" in item_data:
                        page_content = []
                        for section in item_data.get("sections", []):
                            heading_data = section.get("heading")
                            heading_text = heading_data.get("text", "") if isinstance(heading_data, dict) else heading_data
                            page_content.append({
                                "heading": heading_text or None,
                                "paragraphs": section.get("content", [])
                            })
                        transformed_scraped_data_for_ai.append({"url": url, "content": page_content})
                    elif item_type == "raw" and isinstance(item_data, str):
                        # If AI needs raw, store raw. For now, matching find_relevant_content's expectation
                        transformed_scraped_data_for_ai.append({"url": url, "raw_data": item_data})

                relevant_sections, meaningful_match_found = find_relevant_content(transformed_scraped_data_for_ai, user_query)

                if not relevant_sections or not meaningful_match_found:
                    return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response based on the scraped content and your query."})

                prompt_text = f"""As a knowledgeable agent, please provide a direct and conversational answer to the user's question based *only* on the provided website content below.
Do not mention that you are using the provided information.
If the answer is not found in the text, state that you cannot provide a helpful response based on the available information.
User question: "{user_query}"
Website content: """
                content_added = False
                for section in relevant_sections:
                    prompt_text += f"\n--- Content from {section.get('url', 'Unknown URL')} ---\n"
                    if section.get('content'):
                        for sub_section in section.get('content', []):
                            heading = sub_section.get('heading', '') or ""
                            paragraphs = sub_section.get('paragraphs', []) or []
                            if heading:
                                prompt_text += f"Heading: {heading}\n"
                                content_added = True
                            if paragraphs:
                                prompt_text += "\n".join(paragraphs) + "\n"
                                content_added = True
                    elif section.get('raw_data'):
                        prompt_text += section['raw_data'] + "\n"
                        content_added = True
                    prompt_text += "--- End of Content ---\n"

                if not content_added:
                    return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response due to an issue processing the scraped content."})

                ai_response = ask_llama(prompt_text)
                return jsonify({"status": "success", "ai_response": ai_response})
            else:
                error_message = scrapy_result.get("error", "Unknown error during Scrapy execution.") if scrapy_result else "Scrapy did not return a valid result."
                logger.error(f"Scrapy execution failed for /scrape (type: {content_type}): {error_message}")
                return jsonify({"status": "error", "error": error_message}), 500

        # --- Handle Crawl Types ('crawl_raw', 'crawl_beautify', 'crawl_ai') ---
        elif content_type in ['crawl_raw', 'crawl_beautify', 'crawl_ai']:
            scrape_mode = 'raw' if content_type == 'crawl_raw' else 'beautify'
            max_pages_to_crawl = int(data.get('max_pages', 50)) # Allow user to specify, default 50
            
            # Scrapy handles the crawling logic internally, so we just run the spider with crawl_enabled
            scrapy_result = run_scrapy_spider(
                urls_str=urls_str_for_spider,
                scrape_mode=scrape_mode,
                crawl_enabled=True, # Enable crawling
                max_pages=max_pages_to_crawl,
                unique_code=unique_code
            )

            if scrapy_result and scrapy_result.get("status") == "success":
                scraped_items = scrapy_result.get("results", [])
                
                if content_type == 'crawl_ai':
                    if not user_query:
                        return jsonify({"status": "error", "error": "user_query is required for 'crawl_ai' type"}), 400

                    # Transform Scrapy output to match file1's structure for find_relevant_content
                    transformed_scraped_data_for_ai = []
                    for item in scraped_items:
                        url = item.get("url", "Unknown URL")
                        item_type = item.get("type")
                        item_data = item.get("data")

                        if item_type == "beautify" and isinstance(item_data, dict) and "sections" in item_data:
                            page_content = []
                            for section in item_data.get("sections", []):
                                heading_data = section.get("heading")
                                heading_text = heading_data.get("text", "") if isinstance(heading_data, dict) else heading_data
                                page_content.append({
                                    "heading": heading_text or None,
                                    "paragraphs": section.get("content", [])
                                })
                            transformed_scraped_data_for_ai.append({"url": url, "content": page_content})
                        elif item_type == "raw" and isinstance(item_data, str):
                            transformed_scraped_data_for_ai.append({"url": url, "raw_data": item_data})

                    relevant_sections, meaningful_match_found = find_relevant_content(transformed_scraped_data_for_ai, user_query)

                    if not relevant_sections or not meaningful_match_found:
                        return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response based on the crawled content and your query."})
                    
                    prompt_text = f"""As a knowledgeable agent, please provide a direct and conversational answer to the user's question based *only* on the provided website content below.
Do not mention that you are using the provided information.
If the answer is not found in the text, state that you cannot provide a helpful response based on the available information.
User question: "{user_query}"
Website content: """
                    content_added = False
                    for section in relevant_sections:
                        prompt_text += f"\n--- Content from {section.get('url', 'Unknown URL')} ---\n"
                        if section.get('content'):
                            for sub_section in section.get('content', []):
                                heading = sub_section.get('heading', '') or ""
                                paragraphs = sub_section.get('paragraphs', []) or []
                                if heading:
                                    prompt_text += f"Heading: {heading}\n"
                                    content_added = True
                                if paragraphs:
                                    prompt_text += "\n".join(paragraphs) + "\n"
                                    content_added = True
                        elif section.get('raw_data'):
                            prompt_text += section['raw_data'] + "\n"
                            content_added = True
                        prompt_text += "--- End of Content ---\n"

                    if not content_added:
                        return jsonify({"status": "success", "ai_response": "I cannot provide a helpful response due to an issue processing the crawled content."})

                    ai_response = ask_llama(prompt_text)
                    return jsonify({"status": "success", "ai_response": ai_response})
                else: # crawl_raw, crawl_beautify
                    # For non-AI crawls, just return the raw/beautified scraped items
                    formatted_results = []
                    for item in scraped_items:
                        if item.get("type") == "beautify" and "data" in item:
                            page_content = []
                            for section in item["data"].get("sections", []):
                                heading_data = section.get("heading")
                                heading_text = heading_data.get("text", "") if isinstance(heading_data, dict) else heading_data
                                page_content.append({
                                    "heading": heading_text or None,
                                    "paragraphs": section.get("content", [])
                                })
                            formatted_results.append({"url": item.get("url"), "content": page_content})
                        elif item.get("type") == "raw" and "data" in item:
                            formatted_results.append({"url": item.get("url"), "raw_data": item.get("data")})
                    
                    return jsonify({"status": "success", "results": formatted_results})
            else:
                error_message = scrapy_result.get("error", "Unknown error during Scrapy execution.") if scrapy_result else "Scrapy did not return a valid result."
                logger.error(f"Scrapy execution failed for /scrape (type: {content_type}): {error_message}")
                return jsonify({"status": "error", "error": error_message}), 500
        else:
            return jsonify({"status": "error", "error": f"Invalid content_type: {content_type}"}), 400

    except Exception as e:
        error_message = f"Internal server error in /scrape: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return jsonify({"status": "error", "error": "An internal server error occurred"}), 500


@app.route('/delete_agent/<unique_code>', methods=['DELETE'])
@requires_auth
def delete_agent(unique_code):
    """Deletes a stored agent file by its unique code."""
    logger.info(f"Request to delete agent for code: {unique_code}")
    file_path = os.path.join(SCRAPED_DATA_DIR, f"{unique_code}.json") # Assume .json extension now

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Successfully deleted agent file: {file_path}")
            return jsonify({"status": "success", "message": f"Agent with unique_code {unique_code} deleted."})
        else:
            logger.warning(f"Attempted to delete non-existent agent file for code: {unique_code}")
            return jsonify({"status": "error", "error": f"Agent with unique_code {unique_code} not found."}), 404
    except OSError as e:
        error_message = f"Error deleting agent data file {unique_code}: {e}"
        logger.error(error_message)
        return jsonify({"status": "error", "error": "Could not delete agent data file"}), 500
    except Exception as e:
         error_message = f"Unexpected error during agent deletion {unique_code}: {e}\n{traceback.format_exc()}"
         logger.error(error_message)
         return jsonify({"status": "error", "error": "An unexpected error occurred during deletion"}), 500


@app.route('/get_stored_file/<unique_code>', methods=['GET'])
@requires_auth
def get_stored_file(unique_code):
    """Retrieves the full content of a stored agent file."""
    logger.info(f"Request to get stored file for code: {unique_code}")
    content = get_stored_content(unique_code) # This now gets the full object
    if content:
        return jsonify({"status": "success", "unique_code": unique_code, "content": content}) # Return the full object
    else:
        logger.warning(f"Stored file not found for code: {unique_code}")
        return jsonify({"status": "error", "error": f"Content not found for unique_code: {unique_code}"}), 404


# --- Main Execution ---
if __name__ == '__main__':
    print(f"Starting Flask server on host 0.0.0.0 port 5000")
    print(f"Serving scraped data from: {os.path.abspath(SCRAPED_DATA_DIR)}")
    # Use waitress or gunicorn for production deployment
    app.run(host='0.0.0.0', port=5000, debug=False) # Set debug=True for development if needed
