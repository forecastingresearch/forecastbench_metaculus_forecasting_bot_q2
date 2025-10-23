import asyncio
from typing import List, Dict
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FastContentExtractor import FastContentExtractor
from prompts import INITIAL_SEARCH_PROMPT, CONTINUATION_SEARCH_PROMPT
import dateparser
from dotenv import load_dotenv
import json
import os
from aiohttp import ClientSession, ClientTimeout
from asknews_sdk import AskNewsSDK
from prompts import context
from dotenv import load_dotenv
import aiohttp
import re
import random
import time
from openai import OpenAI   
import traceback
from src.helpers import keys
load_dotenv()

def get_serper_key() -> str:
    key = os.getenv("GOOGLE_SERPER_API_KEY")
    if key:
        return key
    return keys.get_secret_that_may_not_exist("serper") or ""

def get_asknews_client_id() -> str:
    key = os.getenv("ASKNEWS_CLIENT_ID")
    if key:
        return key
    return keys.get_secret_that_may_not_exist("asknews-key-id") or ""

def get_asknews_secret() -> str:
    key = os.getenv("ASKNEWS_SECRET")
    if key:
        return key
    return keys.get_secret_that_may_not_exist("asknews-key") or ""

def get_perplexity_key() -> str:
    key = os.getenv("PERPLEXITY_API_KEY")
    if key:
        return key
    return keys.get_secret_that_may_not_exist("perplexity") or ""

def get_metaculus_token() -> str:
    key = os.getenv("METACULUS_TOKEN")
    if key:
        return key
    return keys.get_secret_that_may_not_exist("metaculus-token") or ""

def get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    return keys.get_secret_that_may_not_exist("openai") or ""

ASKNEWS_CACHE_ENABLED = True
ASKNEWS_CACHE_PATH = "/tmp/asknews_cache.json"

try:
    with open(ASKNEWS_CACHE_PATH, "r", encoding="utf-8") as _f:
        _asknews_cache: Dict[str, str] = json.load(_f)
except Exception:
    _asknews_cache = {}

def _cache_get(key: str, *, is_dataset: bool) -> str | None:
    if not (ASKNEWS_CACHE_ENABLED and is_dataset):
        return None
    return _asknews_cache.get(key)

def _cache_put(key: str, value: str, *, is_dataset: bool) -> None:
    if not (ASKNEWS_CACHE_ENABLED and is_dataset):
        return
    _asknews_cache[key] = value
    try:
        with open(ASKNEWS_CACHE_PATH, "w", encoding="utf-8") as _f:
            json.dump(_asknews_cache, _f)
    except Exception as e:
        write(f"[AskNews cache] Failed to cache: {e}")

assistant_prompt = """

You are an assistant to a superforecaster and your task involves high-quality information retrieval to help the forecaster make the most informed forecasts. Forecasting involves parsing through an immense trove of internet articles and web content. To make this easier for the forecaster, you read entire articles and extract the key pieces of the articles relevant to the question. The key pieces generally include:

1. Facts, statistics and other objective measurements described in the article
2. Opinions from reliable and named sources (e.g. if the article writes 'according to a 2023 poll by Gallup' or 'The 2025 presidential approval rating poll by Reuters' etc.)
3. Potentially useful opinions from less reliable/not-named sources (you explicitly document the less reliable origins of these opinions though)

Today, you're focusing on the question:

{title}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}

Background information:
{background}

Article to summarize:
{article}

Note: If the web content extraction is incomplete or you believe the quality of the extracted content isn't the best, feel free to add a disclaimer before your summary.

Please summarize only the article given, not injecting your own knowledge or providing a forecast. Aim to achieve a balance between a superficial summary and an overly verbose account. 

"""

def write(x):
    print(x)

def parse_date(date_str: str) -> str:
    parsed_date = dateparser.parse(date_str, settings={'STRICT_PARSING': False})
    if parsed_date:
        return parsed_date.strftime("%b %d, %Y")
    return "Unknown"

def validate_time(before_date_str, source_date_str):
    if source_date_str == "Unknown":
        return False
    before_date = dateparser.parse(before_date_str)
    source_date = dateparser.parse(source_date_str)
    return source_date <= before_date

# new helper: takes raw article text + the question_details dict
async def summarize_article(article: str, question_details: dict) -> str:
    prompt = assistant_prompt.format(
        title=question_details["title"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        background=question_details["description"],
        article=article
    )
    return await call_gpt(prompt)


async def call_asknews(question: str, stage: str, question_details: dict) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search

    Args
    question (str): The raw search string to send to AskNews.
    stage (str): The stage of the forecasting pipeline (either "historical" or "current").
                    Used for logging and caching to distinguish between context phases.
    question_details (dict): Metadata for the current question (e.g., question_set_name,
                                question_id, source, etc.), used to include identifying info
                                in logs and cache keys.

    Returns
    str: The formatted AskNews response or a cached result if available.

    """
    try:
        if not bool(question_details.get("is_dataset")):
            return "AskNews lookup skipped for non-dataset question."
        ask = AskNewsSDK(
            client_id=get_asknews_client_id(), client_secret=get_asknews_secret(), scopes=set(["news"])
        )
        qid = question_details.get("id")
        qset = question_details.get("question_set")
        qsrc = question_details.get("source")
        is_dataset = bool(question_details.get("is_dataset"))
        qres = question_details.get("resolution_date")
        cache_key = f"{qid}::{question}".strip()

        cached = _cache_get(cache_key, is_dataset=is_dataset)
        if cached is not None:
            print(f"[AskNews] cache_hit stage={stage} set={qset} source={qsrc} id={qid} query={json.dumps({'q':question})}")
            return cached

        payload_latest = {"query": question, "n_articles": 8, "return_type": "both", "strategy": "latest news"}
        payload_knowledge = {"query": question, "n_articles": 8, "return_type": "both", "strategy": "news knowledge"}
        print(f"[AskNews] request stage={stage} set={qset} source={qsrc} id={qid} res_date={qres} payload={json.dumps(payload_latest)}")
        print(f"[AskNews] request stage={stage} set={qset} source={qsrc} id={qid} res_date={qres} payload={json.dumps(payload_knowledge)}")

        async with aiohttp.ClientSession() as session:
            # Create tasks for both API calls
            hot_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
                query=question,
                n_articles=8,
                return_type="both",
                strategy="latest news"
            ))
            historical_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
                query=question,
                n_articles=8,
                return_type="both",
                strategy="news knowledge"
            ))

            # Wait for both tasks to complete
            hot_response, historical_response = await asyncio.gather(hot_task, historical_task)

        hot_articles = hot_response.as_dicts
        historical_articles = historical_response.as_dicts
        formatted_articles = "Here are the relevant news articles:\n\n"

        if hot_articles:
            hot_articles = [article.__dict__ for article in hot_articles]
            hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

            for article in hot_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if historical_articles:
            historical_articles = [article.__dict__ for article in historical_articles]
            historical_articles = sorted(
                historical_articles, key=lambda x: x["pub_date"], reverse=True
            )

            for article in historical_articles:
                pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
                formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

        if not hot_articles and not historical_articles:
            formatted_articles += "No articles were found.\n\n"
            _cache_put(cache_key, formatted_articles,is_dataset=is_dataset)
            return formatted_articles

        _cache_put(cache_key, formatted_articles,is_dataset=is_dataset)
        return formatted_articles

    except Exception as e:
        write(f"[call_asknews] Error: {str(e)}")
        return f"Error retrieving news articles: {str(e)}"
    

async def agentic_search(query: str) -> str:
    """
    Performs agentic search using GPT to iteratively research and analyze a query.
    
    Args:
        query: The search query to research
        
    Returns:
        The final comprehensive analysis
    """
    write(f"[agentic_search] Starting research for query: {query}")
    
    max_steps = 7
    current_analysis = ""
    all_search_queries = []  # Track all queries used
    
    # Cost tracking variables
    total_input_tokens = 0
    total_output_tokens = 0
    
    def estimate_tokens(text: str) -> int:
        """Estimate token count using ~4 characters per token rule for GPT models"""
        return max(1, len(text) // 4)
    
    def calculate_cost(input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        input_cost = (input_tokens / 1_000_000) * 1.100  # $1.100 per 1M input tokens
        output_cost = (output_tokens / 1_000_000) * 4.400  # $4.400 per 1M output tokens
        return input_cost + output_cost
    
    for step in range(max_steps):
        try:
            # Prepare the prompt
            if step == 0:
                prompt = INITIAL_SEARCH_PROMPT.format(query=query)
            else:
                # Build previous section
                if current_analysis:
                    previous_section = f"Your previous analysis:\n{current_analysis}\n\nPrevious search queries used: {', '.join(all_search_queries)}\n"
                else:
                    previous_section = f"Previous search queries used: {', '.join(all_search_queries)}\n"
                
                prompt = CONTINUATION_SEARCH_PROMPT.format(
                    query=query,
                    previous_section=previous_section,
                    search_results=search_results
                )
            
            # Track input tokens
            prompt_tokens = estimate_tokens(prompt)
            total_input_tokens += prompt_tokens
            
            # Call GPT for analysis and search queries
            write(f"[agentic_search] Step {step + 1}: Calling GPT")
            response = await call_gpt(prompt, step)
            
            # Track output tokens
            response_tokens = estimate_tokens(response)
            total_output_tokens += response_tokens
            
            # Parse the response
            analysis_match = re.search(r'Analysis:\s*(.*?)(?=Search queries:|$)', response, re.DOTALL)
            if not analysis_match:
                write(f"[agentic_search] Error: Could not parse analysis from response")
                return f"Error: Failed to parse analysis at step {step + 1}"
            
            # Only update current_analysis after the first search (step > 0)
            if step > 0:
                current_analysis = analysis_match.group(1).strip()
                write(f"[agentic_search] Step {step + 1}: Analysis updated ({len(current_analysis)} chars)")
            else:
                write(f"[agentic_search] Step 1: Initial query understanding complete")
            
            # Check for search queries
            search_queries_match = re.search(r'Search queries:\s*(.*)', response, re.DOTALL)
            
            # For the initial step, we expect search queries
            if step == 0 and not search_queries_match:
                write(f"[agentic_search] Error: No search queries in initial response")
                return "Error: Failed to generate initial search queries"
            
            if not search_queries_match or step == max_steps - 1:
                # No more searches needed or reached max steps
                if step > 0:  # Only break if we have an analysis
                    write(f"[agentic_search] Research complete at step {step + 1}")
                    break
            
            # Extract search queries with sources
            queries_text = search_queries_match.group(1).strip()
            # Parse format: X. [Query] (Source)
            search_queries_with_source = re.findall(r'\d+\.\s*([^(]+?)\s*\((Google|Google News)\)', queries_text)
            
            if not search_queries_with_source:
                if step == 0:
                    write(f"[agentic_search] Error: No valid search queries in initial response")
                    return "Error: Failed to parse initial search queries"
                else:
                    write(f"[agentic_search] No new search queries, completing research")
                    break
            
            # Limit to 5 queries and clean them up
            search_queries_with_source = [(q.strip(), source) for q, source in search_queries_with_source[:5]]
            
            write(f"[agentic_search] Step {step + 1}: Found {len(search_queries_with_source)} search queries")
            # Track just the queries for deduplication
            all_search_queries.extend([q for q, _ in search_queries_with_source])
            
            # Execute searches in parallel
            search_tasks = []
            for sq, source in search_queries_with_source:
                write(f"[agentic_search] Searching: {sq} (Source: {source})")
                search_tasks.append(
                    google_search_agentic(
                        sq,
                        is_news=(source == "Google News")
                    )
                )
            
            # Gather search results
            search_results_list = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Format search results
            search_results = ""
            for (sq, source), result in zip(search_queries_with_source, search_results_list):
                if isinstance(result, Exception):
                    search_results += f"\nSearch query: {sq} (Source: {source})\nError: {str(result)}\n"
                else:
                    search_results += f"\nSearch query: {sq} (Source: {source})\n{result}\n"
            
            write(f"[agentic_search] Step {step + 1}: Search complete, {len(search_results)} chars of results")
            
        except Exception as e:
            write(f"[agentic_search] Error at step {step + 1}: {str(e)}")
            if current_analysis:
                # Return what we have so far
                break
            else:
                return f"Error during agentic search: {str(e)}"
    
    # Print summary statistics
    steps_used = step + 1
    total_cost = calculate_cost(total_input_tokens, total_output_tokens)
    
    print(f"\n🔍 Agentic Search Summary:")
    print(f"   Steps used: {steps_used}")
    print(f"   Total tokens: {total_input_tokens + total_output_tokens:,} ({total_input_tokens:,} input + {total_output_tokens:,} output)")
    print(f"   Estimated cost: ${total_cost:.4f}")
    
    # Ensure we have an analysis to return
    if not current_analysis:
        return "Error: No analysis was generated during the research process"
    
    return current_analysis


async def call_perplexity(prompt: str) -> str:
    """
    Async function to call Perplexity API for deep research
    Includes retry logic and proper timeout handling
    """
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar-deep-research",
        "messages": [
            {
                "role": "system",
                "content": "Be thorough and detailed. Be objective in your analysis, proving documented facts only. Cite all sources with names and dates."
            },
            {
                "role": "user",
                "content": prompt + " Cite all sources with names and dates, compiling a list of sources at the end. Be objective in your analysis, providing documented facts only."
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {get_perplexity_key()}"
    }

    max_retries = 3
    backoff_base = 3  # seconds to wait between retries

    for attempt in range(1, max_retries + 1):
        try:
            write(f"[Perplexity API] Attempt {attempt} for query: {prompt[:50]}...")
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=800)  # 800 seconds timeout
                async with session.post(url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                        write(f"[Perplexity API] ✅ Success on attempt {attempt}")
                        return content.strip()
                    else:
                        response_text = await response.text()
                        write(f"[Perplexity API] ❌ Error: HTTP {response.status}: {response_text}")
                        # Continue to retry on non-200 response
                        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            write(f"[Perplexity API] ⚠️ Attempt {attempt} failed: {e}")
        
        # Only enter retry logic if not on last attempt
        if attempt < max_retries:
            wait_time = backoff_base * attempt
            write(f"[Perplexity API] 🔁 Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
        else:
            write(f"[Perplexity API] ❌ Max retries ({max_retries}) reached. Giving up.")
            return f"Error: Perplexity API failed after {max_retries} attempts. The system will continue with other available data."

    # Should never reach here
    return "Unexpected error in call_perplexity"

async def google_search(query, is_news=False, date_before=None):
    original_query = query
    query = query.replace('"', '').replace("'", '').strip()
    write(f"[google_search] Cleaned query: '{query}' (original: '{original_query}') | is_news={is_news}, date_before={date_before}")
    
    search_type = "news" if is_news else "search"
    url = f"https://google.serper.dev/{search_type}"
    headers = {
        'X-API-KEY': get_serper_key(),
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "q": query,
        "num": 20
    })
    timeout = ClientTimeout(total=70)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('news' if is_news else 'organic', [])
                    write(f"[google_search] Found {len(items)} raw results")

                    filtered_items = []
                    for item in items:
                        item_url = item.get('link')
                        item_date_str = item.get('date', '')
                        item_date = parse_date(item_date_str)
                        if date_before:
                            if item_date != "Unknown" and validate_time(date_before, item_date):
                                write(f"[google_search] ✅ Keeping: {item_url} (date: {item_date})")
                                filtered_items.append(item)
                            else:
                                write(f"[google_search] ❌ Dropped by date: {item_url} (date: {item_date})")
                        else:
                            write(f"[google_search] ✅ Keeping: {item_url}")
                            filtered_items.append(item)

                        if len(filtered_items) >=12:
                            break
                    
                    urls = [item['link'] for item in filtered_items]
                    write(f"[google_search] Returning {len(urls)} URLs: {urls}")
                    return urls
                else:
                    write(f"[google_search] Error in Serper API response: Status {response.status}")
                    response.raise_for_status()
    except Exception as e:
        write(f"[google_search] Exception: {str(e)}")
        raise


async def call_gpt(prompt, step=1):
    client = OpenAI(api_key=get_openai_key())

    try:
        response = client.responses.create(
            model="o3",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        write(f"[call_gpt] Error: {str(e)}")
        return f"Error calling OpenAI API: {str(e)}"


async def google_search_and_scrape(query, is_news, question_details, date_before=None):
    write(f"[google_search_and_scrape] Called with query='{query}', is_news={is_news}, date_before={date_before}")
    try:
        urls = await google_search(query, is_news, date_before)

        if not urls:
            write(f"[google_search_and_scrape] ❌ No URLs returned for query: '{query}'")
            return f"<Summary query=\"{query}\">No URLs returned from Google.</Summary>\n"

        async with FastContentExtractor() as extractor:
            write(f"[google_search_and_scrape] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_and_scrape] ✅ Finished content extraction")

        summarize_tasks = []
        no_results = 3
        valid_urls = []
        for url, data in results.items():
            if len(summarize_tasks) >= no_results:
                break  
            content = (data.get('content') or '').strip()
            if len(content.split()) < 100:
                write(f"[google_search_and_scrape] ⚠️ Skipping low-content article: {url}")
                continue
            if content:
                truncated = content[:8000]
                write(f"[google_search_and_scrape] ✂️ Truncated content for summarization: {len(truncated)} chars from {url}")
                summarize_tasks.append(
                    asyncio.create_task(summarize_article(truncated, question_details))
                )
                valid_urls.append(url)
            else:
                write(f"[google_search_and_scrape] ⚠️ No content for {url}, skipping summarization.")

        if not summarize_tasks:
            write("[google_search_and_scrape] ⚠️ Warning: No content to summarize")
            return f"<Summary query=\"{query}\">No usable content extracted from any URL.</Summary>\n"

        summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

        output = ""
        for url, summary in zip(valid_urls, summaries):
            if isinstance(summary, Exception):
                write(f"[google_search_and_scrape] ❌ Error summarizing {url}: {summary}")
                output += f"\n<Summary source=\"{url}\">\nError summarizing content: {str(summary)}\n</Summary>\n"
            else:
                output += f"\n<Summary source=\"{url}\">\n{summary}\n</Summary>\n"

        return output
    except Exception as e:
        write(f"[google_search_and_scrape] Error: {str(e)}")
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<Summary query=\"{query}\">Error during search and scrape: {str(e)}</Summary>\n"
    

async def google_search_agentic(query, is_news=False):
    """
    Performs Google search and returns raw article content without summarization.
    Used for agentic search where the agent will analyze the raw content.
    
    Args:
        query: Search query string
        is_news: Whether to search Google News (True) or regular Google (False)
        
    Returns:
        Formatted string with raw article contents
    """
    write(f"[google_search_agentic] Called with query='{query}', is_news={is_news}")
    try:
        urls = await google_search(query, is_news)

        if not urls:
            write(f"[google_search_agentic] ❌ No URLs returned for query: '{query}'")
            return f"<RawContent query=\"{query}\">No URLs returned from Google.</RawContent>\n"

        async with FastContentExtractor() as extractor:
            write(f"[google_search_agentic] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_agentic] ✅ Finished content extraction")

        output = ""
        no_results = 3
        results_count = 0
        
        for url, data in results.items():
            if results_count >= no_results:
                break
                
            content = (data.get('content') or '').strip()
            if len(content.split()) < 100:
                write(f"[google_search_agentic] ⚠️ Skipping low-content article: {url}")
                continue
                
            if content:
                truncated = content[:8000]
                write(f"[google_search_agentic] ✂️ Including content: {len(truncated)} chars from {url}")
                output += f"\n<RawContent source=\"{url}\">\n{truncated}\n</RawContent>\n"
                results_count += 1
            else:
                write(f"[google_search_agentic] ⚠️ No content for {url}, skipping.")

        if not output:
            write("[google_search_agentic] ⚠️ Warning: No usable content found")
            return f"<RawContent query=\"{query}\">No usable content extracted from any URL.</RawContent>\n"

        return output
        
    except Exception as e:
        write(f"[google_search_agentic] Error: {str(e)}")
        import traceback
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<RawContent query=\"{query}\">Error during search: {str(e)}</RawContent>\n"



async def process_search_queries(response: str, forecaster_id: str, question_details: dict):
    """
    Parses out search queries from the forecaster's response, executes them
    (AskNews, Agent or Google/Google News), and returns formatted summaries.
    Note: Agent replaces the previous Perplexity functionality.
    """
    try:
        # 1) Extract the "Search queries:" block
        search_queries_block = re.search(r'(?:Search queries:)(.*)', response, re.DOTALL | re.IGNORECASE)
        if not search_queries_block:
            write(f"Forecaster {forecaster_id}: No search queries block found")
            return ""

        queries_text = search_queries_block.group(1).strip()

        # 2) Try to find queries of the form: 1. "text" (Source)
        # Support both "Perplexity" (legacy) and "Agent" (new)
        search_queries = re.findall(
            r'(?:\d+\.\s*)?(["\']?(.*?)["\']?)\s*\((Google|Google News|Assistant|Agent|Perplexity)\)',
            queries_text
        )
        # 3) Fallback to unquoted queries if none found
        if not search_queries:
            search_queries = re.findall(
                r'(?:\d+\.\s*)?([^(\n]+)\s*\((Google|Google News|Assistant|Agent|Perplexity)\)',
                queries_text
            )

        if not search_queries:
            write(f"Forecaster {forecaster_id}: No valid search queries found:\n{queries_text}")
            return ""

        write(f"Forecaster {forecaster_id}: Processing {len(search_queries)} search queries")

        # 4) Kick off one asyncio task per query
        tasks = []
        query_sources = []  # Track which source goes with which task
        stage = "historical" if forecaster_id == "-1" else ("current" if forecaster_id == "0" else f"f{forecaster_id}")
        
        for match in search_queries:
            # match can be ("\"text\"", "text", "Source") or ("text", "Source")
            if len(match) == 3:
                _, raw_query, source = match
            else:
                raw_query, source = match

            query = raw_query.strip().strip('"').strip("'")
            if not query:
                continue

            # Map Perplexity to Agent for backward compatibility
            if source == "Perplexity":
                source = "Agent"
                write(f"Forecaster {forecaster_id}: Mapping Perplexity → Agent for query='{query}'")
            
            write(f"Forecaster {forecaster_id}: Query='{query}' Source={source}")
            query_sources.append((query, source))

            if source in ("Google", "Google News"):
                # pass question_details through so summarizer can fill the prompt
                tasks.append(
                    google_search_and_scrape(
                        query,
                        is_news=(source == "Google News"),
                        question_details=question_details,
                        date_before=question_details.get("resolution_date")
                    )
                )
            elif source == "Assistant":
                if bool(question_details.get("is_dataset")):
                    tasks.append(call_asknews(query, stage, question_details))
                else:
                    write(f"Forecaster {forecaster_id}: Skipping AskNews for market question")
            elif source == "Agent":
                tasks.append(agentic_search(query))

        if not tasks:
            write(f"Forecaster {forecaster_id}: No tasks generated")
            return ""

        # 5) Await all tasks
        formatted_results = ""
        
        # First gather with return_exceptions=True to prevent one failure from breaking everything
        results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # 6) Format the outputs
        for (query, source), result in zip(query_sources, results):
            if isinstance(result, Exception):
                write(f"[process_search_queries] ❌ Forecaster {forecaster_id}: Error for '{query}' → {str(result)}")
                # Add a message about the error in the formatted results
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\nError retrieving results: {str(result)}\n</Asknews_articles>\n"
                elif source == "Agent":
                    formatted_results += f"\n<Agent_report>\nQuery: {query}\n{result}\n</Agent_report>\n"
                else:
                    formatted_results += f"\n<Summary query=\"{query}\">\nError retrieving results: {str(result)}\n</Summary>\n"
            else:
                write(f"[process_search_queries] ✅ Forecaster {forecaster_id}: Query '{query}' processed successfully.")
                
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\n{result}</Asknews_articles>\n"
                elif source == "Agent":
                    formatted_results += f"\n<Agent_report>\nQuery: {query}\n{result}</Agent_report>\n"
                else:
                    # Google/Google News tasks already return <Summary> blocks
                    formatted_results += result

        return formatted_results

    except Exception as e:
        write(f"Forecaster {forecaster_id}: Error processing search queries: {str(e)}")
        import traceback
        write(f"Traceback: {traceback.format_exc()}")
        # Return what we have so far instead of nothing
        return "Error processing some search queries. Partial results may be available."


async def main():
    """
    Demonstrates the usage of process_search_queries with sample search queries.
    """
    print("Starting test for content extraction...")
    
    # This part won't be executed
    sample_response = """
    Search queries:
    1. "Nvidia stock price forecast 2024" (Google)
    2. "Ukraine Russia conflict latest developments" (Google News)
    3. "Middle East stability assessment Israel Hamas" (Perplexity)
    4. "Trump tariffs economic impact" (Assistant)
    """
    
    forecaster_id = "demo_forecaster"
    print(f"Processing sample search queries for forecaster: {forecaster_id}")
    
    # Sample question_details dict
    question_details = {
        "title": "Sample Question",
        "resolution_criteria": "Sample resolution criteria",
        "fine_print": "Sample fine print",
        "description": "Sample background information",
        "resolution_date": "2025-12-31"
    }
    
    results = await process_search_queries(sample_response, forecaster_id, question_details)
    
    print("\n=== SEARCH RESULTS ===\n")
    print(results)
    print("\n=== END OF RESULTS ===\n")

if __name__ == "__main__":
    asyncio.run(main())