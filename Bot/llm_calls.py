import asyncio
import numpy as np
import os
from aiohttp import ClientSession, ClientTimeout, ClientError
import json
import sys
from openai import OpenAI
from anthropic import Anthropic
import re
import io
from dotenv import load_dotenv
from prompts import claude_context, gpt_context
"""
This file contains the main forecasting logic, question-type specific functions are abstracted.
"""
def write(x):
    print(x)

load_dotenv()
from src.helpers import keys

def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY") or keys.get_secret_that_may_not_exist("openai") or ""

def get_anthropic_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY") or keys.get_secret_that_may_not_exist("anthropic") or ""


async def call_anthropic_api(prompt, max_tokens=16000, max_retries=7, cached_content=claude_context):
    client = Anthropic(api_key=get_anthropic_key())
    for attempt in range(max_retries):
        backoff_delay = min(2 ** attempt, 60)
        try:
            write(f"Starting API call attempt {attempt + 1}")

            resp = await asyncio.to_thread(
                client.messages.create,
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=cached_content,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "enabled", "budget_tokens": 12000},
            )

            text = ""
            thinking = ""
            for block in getattr(resp, "content", []) or []:
                t = getattr(block, "type", None)
                if t == "text":
                    text = getattr(block, "text", "") or text
                elif t == "thinking":
                    thinking = getattr(block, "thinking", "") or thinking

            if text:
                return text

            write("No 'text' block found in content.")
            return "No final answer found in Claude response."

        except (ClientError, asyncio.TimeoutError) as e:
            write(f"Retryable error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_delay)

        except Exception as e:
            write(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(backoff_delay)

    raise Exception(f"Failed after {max_retries} attempts")


async def call_claude(prompt):
    try:
        response = await call_anthropic_api(prompt)
        
        if not response:
            write("Warning: Empty response from Anthropic API")
            return "API returned empty response"
            
        return response
        
    except Exception as e:
        write(f"Error in call_claude: {str(e)}")
        return f"Error generating response: {str(e)}"
    

def extract_and_run_python_code(llm_output: str) -> str:
    pattern = re.compile(r'<python>(.*?)</python>', re.DOTALL)
    matches = pattern.findall(llm_output)

    if not matches:
        return "No <python> block found."

    python_code = matches[0].strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        exec(python_code, {})  # use isolated globals
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error executing the extracted Python code:\n{tb}"
    finally:
        sys.stdout = old_stdout

    return new_stdout.getvalue()

# Calls o4-mini using personal OpenAI credentials
async def call_gpt_o4_mini(prompt):
    client = OpenAI(api_key=get_openai_key())
    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model="o4-mini",
            input= gpt_context + "\n" + prompt
        )
        return response.output_text
    except Exception as e:
        write(f"[call_gpt] Error: {str(e)}")
        return f"Error generating response: {str(e)}"

async def call_gpt_o3(prompt):
    client = OpenAI(api_key=get_openai_key())
    try:
        response = await asyncio.to_thread(
            client.responses.create,
            model="o3",
            input= gpt_context + "\n" + prompt
        )
        return response.output_text
    except Exception as e:
        write(f"Error in call_gpt_o3: {str(e)}")
        return f"Error generating response: {str(e)}"
