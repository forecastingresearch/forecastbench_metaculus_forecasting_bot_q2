import asyncio
import datetime
import re
import numpy as np
from prompts import (
    BINARY_PROMPT_historical,
    BINARY_PROMPT_current,
    BINARY_PROMPT_1,
    BINARY_PROMPT_2,
)
from llm_calls import call_claude, call_gpt_o3, call_gpt_o4_mini
from search import process_search_queries

"""
Program flow:
1. Take BINARY_PROMPT_historical and BINARY_PROMPT_current, format in the title, date, background, resolution criteria, fine print (as below) and run call_claude simultaneously on both prompts 
2. Take the output of call_claude, and run it through process_search_queries (programmed to handle raw LLM output, set forecaster_id = "-1" for historical and "0" for current)
3. Use the output of process_search_queries with forecaster_id = "-1" as context for binary prompt 1
4. Take Binary Prompt 1, format in the title, date, resolution_criteria, fine print and context and run, simultaneously, two instances of call_claude (forecaster_id will be 1 and 2 respectively, all strings), two instances of gpt-o4-mini (forecaster_id will be 3 and 4 respectively, all strings) and one instance of gpt-o3 (forecaster_id will be 5)
5. First, initialize a context dictionary with context[1], context[2], context[3], context[4], context[5], all equal to the result of process_search_queries with forecaster_id = "0" (we got this in step 2). Then, we will process the output of Binary Prompt 1 as follows:
    (a) The output of forecaster_id 1 is appended to context[1]
    (b) The output of forecaster_id 2 is appended to context[3]
    (c) The output of forecaster_id 3 is appended to context[2]
    (d) The output of forecaster_id 4 is appended to context[4]
    (e) The output of forecaster_id 5 is appended to context[5]
6. Now, take binary prompt 2, format in the title, date, resolution_criteria, fine_print and respective context (i.e., forecaster x gets context[x]) and run, simultaneously, all five instances we ran previously
7. Pass the output of all five instances to extract_probability_from_response_as_percentage_not_decimal to extract the five probabilities
8. Average the five probabilities, first four with weight 1 and last (from o3) with weight 2 to get the final probability
9. The output should be the final probabilities and the final outputs of binary prompt 2, clearly indicating which output belongs to which forecaster
"""


def extract_probability_from_response_as_percentage_not_decimal(forecast_text: str) -> float:
    matches = re.findall(r"Probability:\s*([0-9]+(?:\.[0-9]+)?)%", forecast_text.strip())
    if matches:
        number = float(matches[-1])
        return min(99, max(1, number))
    raise ValueError(f"Could not extract prediction from response: {forecast_text}")

def log_prompt(model: str, stage: str, prompt: str, details: dict, write):
    meta = f"set={details.get('question_set')} source={details.get('source')} id={details.get('id')}"
    if details.get("resolution_date"):
        meta += f" res_date={details['resolution_date']}"
    write(f"[LLM] model={model} stage={stage} meta=({meta})\nPROMPT:\n{prompt}\n---")

async def get_binary_forecast(question_details, write=print):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    title = question_details["title"]
    resolution_criteria = question_details["resolution_criteria"]
    background = question_details["description"]
    fine_print = question_details["fine_print"]
    resolution_date = question_details.get("resolution_date")
    market_freeze = question_details.get("market_freeze")

    async def format_and_call_gpt(prompt_template, stage: str):
        content = prompt_template.format(
            title=title,
            today=today,
            background=background,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            resolution_date=resolution_date,
            market_freeze=market_freeze,
        )
        log_prompt("gpt-o3", stage, content, question_details, write)
        return content, await call_gpt_o3(content)

    historical_task = asyncio.create_task(format_and_call_gpt(BINARY_PROMPT_historical, "historical_context"))
    current_task = asyncio.create_task(format_and_call_gpt(BINARY_PROMPT_current, "current_context"))
    (historical_prompt, historical_output), (current_prompt, current_output) = await asyncio.gather(
        historical_task, current_task
    )

    context_historical, context_current = await asyncio.gather(
        process_search_queries(historical_output, forecaster_id="-1", question_details=question_details),
        process_search_queries(current_output, forecaster_id="0", question_details=question_details),
    )

    write("\nHistorical context LLM output:\n" + historical_output)
    write("\nCurrent context LLM output:\n" + current_output)
    write("\nHistorical context search results:\n" + context_historical)
    write("\nCurrent context search results:\n" + context_current)

    prompt1 = BINARY_PROMPT_1.format(
        title=title,
        today=today,
        resolution_criteria=resolution_criteria,
        fine_print=fine_print,
        context=context_historical,
        resolution_date=resolution_date,
        market_freeze=market_freeze,
    )

    async def run_prompt1():
        log_prompt("claude", "prompt1_f1", prompt1, question_details, write)  # changed
        log_prompt("claude", "prompt1_f2", prompt1, question_details, write)  # changed
        log_prompt("gpt-o4-mini", "prompt1_f3", prompt1, question_details, write)  # changed
        log_prompt("gpt-o3", "prompt1_f4", prompt1, question_details, write)  # changed
        log_prompt("gpt-o3", "prompt1_f5", prompt1, question_details, write)  # changed
        return await asyncio.gather(
            call_claude(prompt1),
            call_claude(prompt1),
            call_gpt_o4_mini(prompt1),
            call_gpt_o3(prompt1),
            call_gpt_o3(prompt1),
        )

    results_prompt1 = await run_prompt1()

    for i, res in enumerate(results_prompt1):
        write(f"\nForecaster_{i+1} step 1 output:\n{res}")

    context_map = {
        "1": f"Current context: {context_current}\nOutside view prediction: {results_prompt1[0]}",
        "2": f"Current context: {context_current}\nOutside view prediction: {results_prompt1[2]}",
        "3": f"Current context: {context_current}\nOutside view prediction: {results_prompt1[1]}",
        "4": f"Current context: {context_current}\nInside view prediction: {results_prompt1[3]}",
        "5": f"Current context: {context_current}\nInside view prediction: {results_prompt1[4]}",
    }

    def format_prompt2(f_id: str):
        return BINARY_PROMPT_2.format(
            title=title,
            today=today,
            resolution_criteria=resolution_criteria,
            fine_print=fine_print,
            context=context_map[f_id],
            resolution_date=resolution_date,
            market_freeze=market_freeze,
        )

    async def run_prompt2():
        p1 = format_prompt2("1")
        p2 = format_prompt2("2")
        p3 = format_prompt2("3")
        p4 = format_prompt2("4")
        p5 = format_prompt2("5")
        log_prompt("claude", "prompt2_f1", p1, question_details, write)
        log_prompt("claude", "prompt2_f2", p2, question_details, write)
        log_prompt("gpt-o4-mini", "prompt2_f3", p3, question_details, write)
        log_prompt("gpt-o3", "prompt2_f4", p4, question_details, write)
        log_prompt("gpt-o3", "prompt2_f5", p5, question_details, write)
        return await asyncio.gather(
            call_claude(p1),
            call_claude(p2),
            call_gpt_o4_mini(p3),
            call_gpt_o3(p4),
            call_gpt_o3(p5),
        )
    results_prompt2 = await run_prompt2()

    probabilities = []
    for r in results_prompt2:
        try:
            prob = extract_probability_from_response_as_percentage_not_decimal(r)
            probabilities.append(prob)
        except Exception as e:
            write(f"Error extracting probability: {e}")
            probabilities.append(None)

    valid_probs = [p for p in probabilities if p is not None]
    if len(valid_probs) >= 1:
        weights = [1, 1, 1, 2, 2]  # forecaster 5 (o3) has truple weight
        weighted_probs = [p * w for p, w in zip(probabilities, weights) if p is not None]
        weight_sum = sum(w for p, w in zip(probabilities, weights) if p is not None)
        final_prob = float(np.sum(weighted_probs) / weight_sum)
        final_prob = min(0.999, max(0.001, final_prob / 100))  # Normalize to [0.001, 0.999]
    else:
        final_prob = None

    write(f"\nFinal predictions: {probabilities}")
    write(f"Result: {final_prob}")

    final_outputs = "\n\n".join(
        f"=== Forecaster {i+1} ===\nOutput:\n{out}\nPredicted Probability: {prob if prob is not None else 'N/A'}%"
        for i, (out, prob) in enumerate(zip(results_prompt2, probabilities))
    )

    write(final_outputs)

    return final_prob, final_outputs
