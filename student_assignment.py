#coding=big5

import json
import requests
import base64

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from mimetypes import guess_type

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

class Festival(BaseModel):
    date: str = Field(description="festival date")
    name: str = Field(description="festival name")

class FestivalResult(BaseModel):
    Result: List[Festival]

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return base64_encoded_data

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    parser = JsonOutputParser(pydantic_object=FestivalResult)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser
    result = chain.invoke({"query": question})

    result = json.dumps(result, ensure_ascii=False, indent=4)

    return result
    
def generate_hw02(question):
    # Refer to
    # https://medium.com/@shravankoninti/agent-tools-basic-code-using-langchain-50e13eb07d92
    # https://python.langchain.com/docs/how_to/function_calling/#passing-tool-outputs-to-model
    # https://calendarific.com/api-documentation

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    prompt = f"""Extract the country, year, month and text's language from the following text:
        Text: "{question}"
        About the country and text's language, please provide the codes according to ISO 3166-1.
        About the month, it should be provided in Arabic numerals.
        Please extract the country code, year, month and language code in the following format:
        country code, year, month, language code
        """

    message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
            ]
    )

    response = llm.invoke([message])

    parts = response.content.split(", ")

    if len(parts) == 4:
        country = parts[0]
        year = parts[1]
        month = parts[2]
        language = parts[3]

        # Calendarific API
        url = 'https://calendarific.com/api/v2/holidays?'

        parameters = {
            # Required
            'country': country,
            'language': language,
            'year':    year,
            'month': month,
            'api_key': 'zZFMAithtEf03ZFRYaa9BRSNDZdX62fF'
        }

        response = requests.get(url, params=parameters)

        # print(response.text)

        result = json.loads(response.text)

        if response.status_code != 200:
            print("Calendarific API return error code: " + response.status_code)
        else:
            holidays = []
            count = len(result["response"]["holidays"])
            for i in range(count):
                holiday_name = result["response"]["holidays"][i]['name']
                holiday_date = result["response"]["holidays"][i]['date']['iso']
                # print(holiday_name)
                # print(holiday_date)

                holidays.append({"date": holiday_date, "name": holiday_name})

            # print(holidays)
            result = {"Result": holidays}

            json_output = json.dumps(result, ensure_ascii=False, indent=4)

            prompt = f"""Translate the text to {language} but the text should be translated to trandition chinese if {country} is Taiwan:
                Text: "{json_output}"
                Return the JSON format directly and don't add "```json" and ```.
                """
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                ]
            )

            response = llm.invoke([message])
            return response.content

    else:
        print("llm return wrong format")

    return

    
def generate_hw03(question2, question3):
    hw2_results = generate_hw02(question2)

    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    prompt = ChatPromptTemplate.from_messages([
        ("ai", "{holiday_list}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    holiday_add_result = chain_with_history.invoke(
        {"holiday_list": hw2_results,
         "question": question3 + "，比對date跟name，判斷該節日是否存在於清單中，如果不存在, 請回答true, 反之則回答false"},
        config={"configurable": {"session_id": "holiday_questions"}}
    )
    # print(holiday_add_result.content)

    holiday_add_bool_result = True
    if (holiday_add_result.content == "false"):
        holiday_add_bool_result = False

    holiday_add_reason = chain_with_history.invoke(
        {"holiday_list": hw2_results,
         "question": f"依照之前回答結果{holiday_add_bool_result}，若為true，表示需要加入清單，反之亦然，請用一行說明需新增甚麼節日並解釋原因，並且列出目前甚麼月份已存在的甚麼節日名稱"},
        config={"configurable": {"session_id": "holiday_questions"}}
    )
    # print(holiday_add_reason.content)

    holiday_result = {"add": holiday_add_bool_result, "reason":f"{holiday_add_reason.content}"}
    result = {"Result": holiday_result}

    json_output = json.dumps(result, ensure_ascii=False, indent=4)

    return json_output
    
def generate_hw04(question):
    # Refer to https://python.langchain.com/docs/how_to/multimodal_inputs/
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    image_data = local_image_to_data_url("./baseball.png")

    # image_data = base64.b64encode(httpx.get("https://www.bing.com/images/blob?bcid=r.x2NZ7VTvYHCg").content).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": f"Reference the image and answer the question: {question} without any explanations"},
            { "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    )

    response = llm.invoke([message])

    score_result = {"score": int(response.content)}
    result = {"Result": score_result}

    json_output = json.dumps(result, ensure_ascii=False, indent=4)

    return json_output
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )

    response = llm.invoke([message])

    return response

# if __name__ == "__main__":
#     result = demo('2024年台灣10月紀念日有哪些?')
#     print(result.content)
#     print('--------------------------------------------')

#     print('--------------------hw01 result------------------------')
#     result = generate_hw01('2024年台灣10月紀念日有哪些?')
#     print(result)

#     print('--------------------hw02 result------------------------')
#     result = generate_hw02('2024年台灣10月紀念日有哪些?')
#     print(result)

#     print('--------------------hw03 result------------------------')
#     result = generate_hw03('2024年台灣10月紀念日有哪些?', '根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？')
#     print(result)

#     print('--------------------hw04 result------------------------')
#     result = generate_hw04('請問中華台北的積分是多少?')
#     print(result)
