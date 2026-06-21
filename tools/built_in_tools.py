## DuckDuckGo Search
from langchain_community.tools import DuckDuckGoSearchRun
search_tool=DuckDuckGoSearchRun()
results=search_tool.invoke('top news in india today')
# print(results)

## Shell Tool
from langchain_community.tools import ShellTool
shell_tool=ShellTool()
result=shell_tool.invoke('whoami')
# print(result)
