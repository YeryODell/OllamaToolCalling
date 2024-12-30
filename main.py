from datetime import datetime
from geopy.geocoders import Nominatim
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from timezonefinder import TimezoneFinder
import logging
import pytz



# logging.basicConfig(level=logging.INFO)

llm = ChatOllama(model="llama3.2:3b")


@tool
def get_day_of_the_week(location: str) -> str:
    """Get the correct day of the week for a specific location."""
    logging.info("'get_day_of_the_week' function was called")
    loc = Nominatim(user_agent="GetLoc")
    getLoc = loc.geocode(location)
    latitude = getLoc.latitude
    longitude = getLoc.longitude
    tz = TimezoneFinder()
    timezone = tz.timezone_at(lng=longitude,lat=latitude)
    return f"{location}: {datetime.now(pytz.timezone(timezone)).strftime("%A")}"


@tool
def get_location_time(location: str) -> str:
    """Get the current time for a specific location."""
    logging.info("'get_location_time' function was called")
    loc = Nominatim(user_agent="GetLoc")
    getLoc = loc.geocode(location)
    latitude = getLoc.latitude
    longitude = getLoc.longitude
    tz = TimezoneFinder()
    timezone = tz.timezone_at(lng=longitude,lat=latitude)
    return f"{location}: {datetime.now(pytz.timezone(timezone)).strftime("%I:%M %p")}"


tools = [get_day_of_the_week, get_location_time]
llm_tools = llm.bind_tools(tools)

prompt = input("Say Something: ")
messages = [HumanMessage(prompt)]
ai_msg = llm_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"get_day_of_the_week": get_day_of_the_week, "get_location_time": get_location_time}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

# for i in range(0, len(messages)):
#     print("----")
#     print(f"{messages[i].type}: {messages[i]}")

final_response = llm_tools.invoke(messages)

print(final_response.content)