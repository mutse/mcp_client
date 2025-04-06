#!/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from langchain_openai import ChatOpenAI


load_dotenv()


async def run_client():
    with open("mcp.json", "r") as file:
        content = file.read()
    data = json.loads(content)

    model = ChatOpenAI(base_url=os.environ.get("BASE_URL"),
                       api_key=os.environ.get("API_KEY"),
                       model=os.environ.get("TOOL_MODEL"))

    async with MultiServerMCPClient(data) as client:
        agent = create_react_agent(model, client.get_tools())
        response = await agent.ainvoke({"messages": "say hello"})
        return response["messages"][-1].content


if __name__ == "__main__":
    result = asyncio.run(run_client())
    print(result)
