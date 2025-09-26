#!/usr/bin/env python3
"""
Mock VLM server for testing color-narrator without needing actual VLM deployment.
Returns realistic but simple color descriptions for testing.
"""
import asyncio
from aiohttp import web


async def handle_chat_completions(request):
    """Mock the OpenAI-compatible /v1/chat/completions endpoint"""
    data = await request.json()

    # Extract image data if present
    messages = data.get("messages", [])
    has_image = any("image_url" in str(msg) for msg in messages)

    # Generate a simple color description based on the request
    if has_image:
        descriptions = [
            "This image exhibits a subtle green color cast throughout, most noticeable in the midtones and shadows. The toning creates a cool, naturalistic atmosphere while maintaining good contrast and detail.",
            "The image shows a distinctive aqua-blue color grading with complementary warm orange highlights. This split-toning technique creates visual depth and a cinematic quality typical of modern digital processing.",
            "Neutral color balance with minimal tinting detected. The image maintains accurate skin tones and natural color reproduction with only minor variations within acceptable tolerances.",
        ]

        import random

        description = random.choice(descriptions)
    else:
        description = "No image provided for color analysis."

    response = {
        "id": "chatcmpl-mock-123",
        "object": "chat.completion",
        "created": 1709640000,
        "model": "Qwen2-VL-7B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": description},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    return web.json_response(response)


async def handle_models(request):
    """Mock the /v1/models endpoint"""
    response = {
        "object": "list",
        "data": [
            {
                "id": "Qwen2-VL-7B-Instruct",
                "object": "model",
                "created": 1709640000,
                "owned_by": "mock",
            }
        ],
    }
    return web.json_response(response)


async def init_app():
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_models)
    return app


async def main():
    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8000)
    await site.start()
    print("🤖 Mock VLM server running on http://localhost:8000")
    print("💡 Available endpoints:")
    print("   GET  /v1/models")
    print("   POST /v1/chat/completions")
    print("\n🔄 Ready to serve color-narrator requests...")

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)  # Sleep for an hour
    except KeyboardInterrupt:
        print("\n🛑 Shutting down mock server...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
