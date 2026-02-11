import json

from ACI_AI_Backend.llmtool import Tool
from langchain_openai import ChatOpenAI
import traceback


class LLM:
    def __init__(
        self,
        deploy_method: str,
        model_name: str,
        base_url: str,
        max_regen_tries: int = 10,
    ):
        if deploy_method == "vllm":
            self.model = ChatOpenAI(
                model=model_name,
                base_url=base_url,
                reasoning_effort="medium",
                api_key="",
            )
        else:
            raise ValueError(f"LLM deployment type unsupported: {deploy_method}")

        self.max_regen_tries = max_regen_tries
        self.tools: dict[str, Tool] = {}
        self.tool_prompt = "" 

    def bind_tools(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}
        tools_description = "\n\n".join(str(tool) for tool in tools)
        self.tool_prompt = (
            "system",
            (
                "You may use tools to help you do your tasks\n\n"
                "If you wish to call a tool, respond ONLY with valid JSON in this exact format:\n"
                "{\n"
                '  "tool": "<tool name>",\n'
                '  "args": { "arg_name": value }\n'
                "}\n\n"
                "Rules:\n"
                "- Use ONLY the tools listed below\n"
                "- Tool arguments MUST match parameter names exactly\n"
                "- Omit optional arguments if not needed\n"
                "- Use correct JSON types\n"
                "- Do not add any extra text\n\n"
                "Available tools:\n\n"
                f"{tools_description}"
            )
        )
        print(tools)
        print(self.tool_prompt)

    def invoke(self, messages: list[tuple[str, str]]) -> str:
        messages_all = [self.tool_prompt] + messages

        tries = 0

        while tries < self.max_regen_tries:
            response = self.model.invoke(messages_all)
            content = response.content.strip()

            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                # The answer is the final answer
                print("Returning final answer")
                print(response)
                return content

            # Add assistant answer to messages
            messages_all.append(response)

            tool_name = payload.get("tool")
            args = payload.get("args", {})

            if tool_name not in self.tools:
                messages.append(("system", f"Unknown tool requested: `{tool_name}`\n\n"))
            else:
                print("Called tool: ", tool_name)

                tool = self.tools[tool_name]
                try:
                    # Executes the tool
                    result = tool.func(**args)
                    messages.append(("system", f"Tool `{tool_name}` was called with arguments {args}.\n" f"Tool result:\n{result}\n\n" "Use this result to continue."))
                except:
                    print(traceback.format_exc())
                    messages.append(("system", f"Tool `{tool_name}` was called with arguments {args}.\n" f"Tool execution failed\n\n"))

                tries += 1

        # Generate without tool calling
        messages_all = messages
        response = self.model.invoke(messages)
        return response.content.strip()
