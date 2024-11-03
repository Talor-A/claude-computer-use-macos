import asyncio
import os
import sys
import json
import base64

from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult
from anthropic.types.beta import BetaMessage, BetaMessageParam
from anthropic import APIResponse


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
            print("Assistant:", content_block.get("text"))

    def tool_output_callback(result: ToolResult, tool_use_id: str):
        if result.output:
            print(f"> Tool Output [{tool_use_id}]:", result.output)
        if result.error:
            print(f"!!! Tool Error [{tool_use_id}]:", result.error)
        if result.base64_image:
            os.makedirs("screenshots", exist_ok=True)
            image_data = result.base64_image
            with open(f"screenshots/screenshot_{tool_use_id}.png", "wb") as f:
                f.write(base64.b64decode(image_data))
            print(f"Took screenshot screenshot_{tool_use_id}.png")

    def api_response_callback(response: APIResponse[BetaMessage]):
        content = json.loads(response.text)["content"]
        print("\n---------------\nAPI Response:")

        for item in content:
            if item["type"] == "text":
                print("\n🤖 Assistant:", item["text"])
            elif item["type"] == "tool_use":
                print(f"\n🔧 Tool Use ({item['name']}):")
                print(f"   Input: {item['input']}")

    # Check for command line argument
    initial_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    if initial_input:
        # Add initial input to messages
        messages.append(
            {
                "role": "user",
                "content": initial_input,
            }
        )

        # Process initial input
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
    else:
        print(
            "Starting Claude 'Computer Use' chat session.\nType 'exit' to quit.\nPress Enter after each message."
        )

    while True:
        # Get user input
        try:
            user_input = input("\nYou: ").strip()
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
