"""
MCP Slack Server with LLM Integration - Connects to Slack and responds using Claude/OpenAI
Enhanced with Daily Summary functionality
"""


import os
import asyncio
import logging

import mcp.server.stdio

from services.mcp_server import SlackMCPServer
from core.logging import configure_logging


configure_logging()


async def main():
    """Main entry point"""
    server = SlackMCPServer()
    
    try:
        # Check if running in MCP mode (stdio) or standalone mode
        if len(os.sys.argv) > 1 and os.sys.argv[1] == "--mcp":
            # Run as MCP server with stdio
            logging.info("Starting in MCP mode...")
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                await server.server.run(
                    read_stream,
                    write_stream,
                    server.server.create_initialization_options()
                )
        else:
            # Run as standalone Slack bot
            logging.info("Starting in standalone mode...")
            await server.start()
            
            # Keep running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logging.info("Received shutdown signal")
                await server.stop()
                
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise




if __name__ == "__main__":
    # Required environment variables check
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logging.error("Please set these in your .env file:")
        logging.error("SLACK_BOT_TOKEN=xoxb-your-bot-token")
        logging.error("SLACK_APP_TOKEN=xapp-your-app-token")
        logging.error("ANTHROPIC_API_KEY=your-claude-key (optional)")
        logging.error("OPENAI_API_KEY=your-openai-key (optional)")
        logging.error("GROQ_API_KEY=your-groq-key (optional)")
        exit(1)
    
    # Check if at least one LLM API key is configured
    llm_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]
    if not any(os.getenv(key) for key in llm_keys):
        logging.warning("No LLM API keys configured. Bot will not be able to generate responses.")
        logging.warning("Please set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY")
    
    # Run the server
    asyncio.run(main())

                