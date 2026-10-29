from typing import Literal


def get_formatted_prompt(
    data: dict,
    call_type: Literal[
        "acompletion",
        "completion",
        "embedding",
        "aembedding",
        "image_generation",
        "transcription",
        "atranscription",
        "audio_transcription",  # legacy alias
        "moderation",
        "amoderation",
        "text_completion",
        "atext_completion",
    ],
) -> str:
    """
    Extracts the prompt from the input data based on the call type.

    Returns a string.
    """
    prompt = ""
    if call_type == "acompletion" or call_type == "completion":
        for message in data["messages"]:
            if message.get("content", None) is not None:
                content = message.get("content")
                if isinstance(content, str):
                    prompt += message["content"]
                elif isinstance(content, list):
                    for c in content:
                        if c["type"] == "text":
                            prompt += c["text"]
            if "tool_calls" in message:
                for tool_call in message["tool_calls"] or ():
                    if "function" in tool_call:
                        function_arguments = tool_call["function"]["arguments"]
                        prompt += function_arguments
    elif call_type in ("text_completion", "atext_completion"):
        prompt = data["prompt"]
    elif call_type in ("embedding", "aembedding", "moderation", "amoderation"):
        if isinstance(data["input"], str):
            prompt = data["input"]
        elif isinstance(data["input"], list):
            for m in data["input"]:
                prompt += m
    elif call_type == "image_generation":
        prompt = data["prompt"]
    elif call_type in ("transcription", "atranscription", "audio_transcription"):
        if "prompt" in data:
            prompt = data["prompt"]
    return prompt
