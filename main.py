import asyncio
import os
import sys
import json
import base64
from colorama import init, Fore, Style

from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult
from anthropic.types.beta import BetaMessage, BetaMessageParam
from anthropic import APIResponse

init()  # Initialize colorama


async def main():
    # Set up your Anthropic API key and model
    api_key = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
    if api_key == "YOUR_API_KEY_HERE":
        raise ValueError(
            "Please first set your API key in the ANTHROPIC_API_KEY environment variable"
        )
    provider = APIProvider.ANTHROPIC

    # Initialize messages list
    messages: list[BetaMessageParam] = []

    # Define callbacks (you can customize these)
    def output_callback(content_block):
        if isinstance(content_block, dict) and content_block.get("type") == "text":
            print(
                f"\n{Fore.CYAN}Assistant:{Style.RESET_ALL}", content_block.get("text")
            )

    def tool_output_callback(result: ToolResult, tool_use_id: str):
        if result.output:
            print(f"\n{Fore.GREEN}> Tool Output [{tool_use_id}]:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{result.output}{Style.RESET_ALL}")
        if result.error:
            print(f"\n{Fore.RED}!!! Tool Error [{tool_use_id}]:{Style.RESET_ALL}")
            print(f"{Fore.RED}{result.error}{Style.RESET_ALL}")
        if result.base64_image:
            os.makedirs("screenshots", exist_ok=True)
            image_data = result.base64_image
            with open(f"screenshots/screenshot_{tool_use_id}.png", "wb") as f:
                f.write(base64.b64decode(image_data))
            print(
                f"\n{Fore.YELLOW}📸 Took screenshot: screenshot_{tool_use_id}.png{Style.RESET_ALL}"
            )

    def api_response_callback(response: APIResponse[BetaMessage]):
        content = json.loads(response.text)["content"]
        print(f"\n{Fore.BLUE}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}API Response:{Style.RESET_ALL}")

        for item in content:
            if item["type"] == "text":
                print(f"\n{Fore.CYAN}🤖 Assistant:{Style.RESET_ALL}", item["text"])
            elif item["type"] == "tool_use":
                print(f"\n{Fore.YELLOW}🔧 Tool Use ({item['name']}):{Style.RESET_ALL}")
                print(
                    f"{Fore.WHITE}   Input: {json.dumps(item['input'], indent=2)}{Style.RESET_ALL}"
                )

    print(
        "Starting Claude 'Computer Use' chat session.\nType 'exit' to quit.\nPress Enter after each message."
    )

    while True:
        # Get user input
        try:
            user_input = input(f"\n{Fore.GREEN}You:{Style.RESET_ALL} ").strip()
            if user_input.lower() == "exit":
                print("Ending chat session...")
                break

            if user_input:
                # Add user message to messages list
                messages.append(
                    {
                        "role": "user",
                        "content": user_input,
                    }
                )

                # Run the sampling loop for this interaction
                messages = await sampling_loop(
                    model="claude-3-5-sonnet-20241022",
                    provider=provider,
                    system_prompt_suffix="",
                    messages=messages,
                    output_callback=output_callback,
                    tool_output_callback=tool_output_callback,
                    api_response_callback=api_response_callback,
                    api_key=api_key,
                    only_n_most_recent_images=10,
                    max_tokens=4096,
                    max_retries=3,
                    initial_retry_delay=8,
                )

        except KeyboardInterrupt:
            print("\nEnding chat session...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Encountered Error:\n{e}")
