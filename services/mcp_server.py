
import re
import os
import json
import asyncio
import logging
from typing import List, Any, Dict
from datetime import datetime, timedelta, time

import mcp.types as types
from mcp.server import Server
from mcp.types import Resource, Tool
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from services.llm_client import LLMClient
from models.data_models import SlackMessage
from services.summary_manager import DailySummaryManager


class SlackMCPServer:
    def __init__(self):
        self.app = None
        self.handler = None
        self.server = Server("slack-mcp-server")
        self.recent_messages: List[SlackMessage] = []
        self.max_messages = 50
        self.llm_client = LLMClient(preferred_llm='groq')  
        self.summary_manager = None
        
        # Bot configuration
        self.bot_user_id = None
        self.respond_to_mentions = True
        self.respond_to_dms = True
        self.respond_in_threads = True
        
        # Initialize MCP server tools and resources
        self._setup_mcp_handlers()
        




    def _setup_mcp_handlers(self):
        """Set up MCP server handlers for tools and resources"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available Slack resources"""
            return [
                Resource(
                    uri="slack://messages/recent",
                    name="Recent Slack Messages",
                    description="Get recent messages from Slack channels",
                    mimeType="application/json",
                ),
                Resource(
                    uri="slack://channels/list", 
                    name="Slack Channels",
                    description="List available Slack channels",
                    mimeType="application/json",
                ),
                Resource(
                    uri="slack://bot/status",
                    name="Bot Status",
                    description="Current bot configuration and status",
                    mimeType="application/json",
                ),
                Resource(
                    uri="slack://summary/status",
                    name="Daily Summary Status",
                    description="Status of daily summary functionality",
                    mimeType="application/json",
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read Slack resources"""
            if uri == "slack://messages/recent":
                messages_data = [
                    {
                        "channel": msg.channel,
                        "user": msg.user, 
                        "text": msg.text,
                        "timestamp": msg.timestamp,
                        "thread_ts": msg.thread_ts
                    }
                    for msg in self.recent_messages[-20:]  # Last 20 messages
                ]
                return json.dumps(messages_data, indent=2)
            
            elif uri == "slack://channels/list":
                if self.app:
                    try:
                        result = await self.app.client.conversations_list()
                        channels = [
                            {
                                "id": channel["id"],
                                "name": channel["name"],
                                "is_channel": channel["is_channel"],
                                "is_private": channel["is_private"]
                            }
                            for channel in result["channels"]
                        ]
                        return json.dumps(channels, indent=2)
                    except Exception as e:
                        logging.error(f"Error fetching channels: {e}")
                        return json.dumps({"error": str(e)})
            
            elif uri == "slack://bot/status":
                status = {
                    "bot_user_id": self.bot_user_id,
                    "respond_to_mentions": self.respond_to_mentions,
                    "respond_to_dms": self.respond_to_dms,
                    "respond_in_threads": self.respond_in_threads,
                    "llm_configured": self.llm_client.claude_api_key is not None or self.llm_client.openai_api_key is not None or self.llm_client.groq_api_key is not None,
                    "preferred_llm": self.llm_client.preferred_llm,
                    "total_messages_processed": len(self.recent_messages)
                }
                return json.dumps(status, indent=2)
                
            elif uri == "slack://summary/status":
                if self.summary_manager:
                    status = {
                        "workspace_owner_id": self.summary_manager.workspace_owner_id,
                        "summary_time": str(self.summary_manager.summary_time),
                        "last_summary_date": str(self.summary_manager.last_summary_date) if self.summary_manager.last_summary_date else None,
                        "next_summary_due": self.summary_manager.should_generate_summary()
                    }
                else:
                    status = {"error": "Summary manager not initialized"}
                return json.dumps(status, indent=2)
                        
            return json.dumps({"error": "Resource not found"})

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="send_slack_message",
                    description="Send a message to a Slack channel",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "description": "Channel ID or name to send message to"
                            },
                            "text": {
                                "type": "string",
                                "description": "Message text to send"
                            },
                            "thread_ts": {
                                "type": "string",
                                "description": "Optional timestamp of message to reply to in thread",
                                "optional": True
                            }
                        },
                        "required": ["channel", "text"]
                    }
                ),
                Tool(
                    name="get_channel_history",
                    description="Get message history from a Slack channel",
                    inputSchema={
                        "type": "object", 
                        "properties": {
                            "channel": {
                                "type": "string",
                                "description": "Channel ID or name"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of messages to retrieve (max 100)",
                                "default": 10
                            }
                        },
                        "required": ["channel"]
                    }
                ),
                Tool(
                    name="ask_llm",
                    description="Ask the configured LLM a question with optional context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Question to ask the LLM"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for the question",
                                "optional": True
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="react_to_message",
                    description="Add a reaction to a Slack message",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string", 
                                "description": "Channel ID"
                            },
                            "timestamp": {
                                "type": "string",
                                "description": "Message timestamp"
                            },
                            "emoji": {
                                "type": "string",
                                "description": "Emoji name (without colons, e.g. 'thumbsup')"
                            }
                        },
                        "required": ["channel", "timestamp", "emoji"]
                    }
                ),
                Tool(
                    name="generate_manual_summary",
                    description="Manually trigger daily summary generation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "Date for summary in YYYY-MM-DD format (optional, defaults to yesterday)",
                                "optional": True
                            }
                        }
                    }
                ),
                Tool(
                    name="set_summary_time",
                    description="Set the time for daily summary generation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hour": {
                                "type": "integer",
                                "description": "Hour (0-23)"
                            },
                            "minute": {
                                "type": "integer",
                                "description": "Minute (0-59)",
                                "default": 0
                            }
                        },
                        "required": ["hour"]
                    }
                )
            ]





        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls"""
            try:
                if name == "send_slack_message":
                    channel = arguments["channel"]
                    text = arguments["text"]
                    thread_ts = arguments.get("thread_ts")
                    
                    if self.app:
                        result = await self.app.client.chat_postMessage(
                            channel=channel,
                            text=text,
                            thread_ts=thread_ts
                        )
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({
                                "success": True,
                                "message_ts": result["ts"],
                                "channel": result["channel"]
                            })
                        )]
                    else:
                        return [types.TextContent(
                            type="text", 
                            text=json.dumps({"error": "Slack app not initialized"})
                        )]
                        
                elif name == "get_channel_history":
                    channel = arguments["channel"]
                    limit = min(arguments.get("limit", 10), 100)
                    
                    if self.app:
                        result = await self.app.client.conversations_history(
                            channel=channel,
                            limit=limit
                        )
                        messages = [
                            {
                                "user": msg.get("user", "unknown"),
                                "text": msg.get("text", ""),
                                "ts": msg.get("ts", ""),
                                "type": msg.get("type", "")
                            }
                            for msg in result["messages"]
                        ]
                        return [types.TextContent(
                            type="text",
                            text=json.dumps(messages, indent=2)
                        )]
                    else:
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"error": "Slack app not initialized"})
                        )]
                
                elif name == "ask_llm":
                    question = arguments["question"]
                    context = arguments.get("context", "")
                    
                    response = await self.llm_client.get_response(question, context)
                    return [types.TextContent(
                        type="text",
                        text=response
                    )]
                        
                elif name == "react_to_message":
                    channel = arguments["channel"]
                    timestamp = arguments["timestamp"] 
                    emoji = arguments["emoji"]
                    
                    if self.app:
                        await self.app.client.reactions_add(
                            channel=channel,
                            timestamp=timestamp,
                            name=emoji
                        )
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"success": True})
                        )]
                    else:
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"error": "Slack app not initialized"})
                        )]
                        
                elif name == "generate_manual_summary":
                    if not self.summary_manager:
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"error": "Summary manager not initialized"})
                        )]
                    
                    date_str = arguments.get("date")
                    if date_str:
                        try:
                            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        except ValueError:
                            return [types.TextContent(
                                type="text",
                                text=json.dumps({"error": "Invalid date format. Use YYYY-MM-DD"})
                            )]
                    else:
                        target_date = (datetime.now() - timedelta(days=1)).date()
                    
                    # Generate summary for specified date
                    try:
                        await self.summary_manager.generate_and_send_daily_summary()
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({
                                "success": True,
                                "message": f"Manual summary generated for {target_date}"
                            })
                        )]
                    except Exception as e:
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"error": f"Failed to generate summary: {str(e)}"})
                        )]
                        
                elif name == "set_summary_time":
                    if not self.summary_manager:
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"error": "Summary manager not initialized"})
                        )]
                    
                    hour = arguments["hour"]
                    minute = arguments.get("minute", 0)
                    
                    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                        return [types.TextContent(
                            type="text",
                            text=json.dumps({"error": "Invalid time. Hour must be 0-23, minute must be 0-59"})
                        )]
                    
                    self.summary_manager.summary_time = time(hour, minute)
                    return [types.TextContent(
                        type="text",
                        text=json.dumps({
                            "success": True,
                            "message": f"Summary time set to {hour:02d}:{minute:02d}"
                        })
                    )]
                else:
                        return [types.TextContent(
                        type="text",
                        text=json.dumps({"error": f"Unknown tool: {name}"})
                    )]
                    
            except Exception as e:
                logging.error(f"Error executing tool {name}: {e}")
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)})
                )]





    async def setup_slack_app(self):
        """Initialize Slack app and handlers"""
        bot_token = os.getenv("SLACK_BOT_TOKEN")
        app_token = os.getenv("SLACK_APP_TOKEN")
        
        if not bot_token or not app_token:
            raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set in environment variables")
        
        # Initialize Slack app
        self.app = AsyncApp(token=bot_token)
        self.handler = AsyncSocketModeHandler(self.app, app_token)
        
        # Initialize summary manager
        self.summary_manager = DailySummaryManager(self.app, self.llm_client)
        await self.summary_manager.initialize()
        
        # Get bot user ID
        try:
            auth_result = await self.app.client.auth_test()
            self.bot_user_id = auth_result["user_id"]
            logging.info(f"Bot user ID: {self.bot_user_id}")
        except Exception as e:
            logging.error(f"Failed to get bot user ID: {e}")
        
        # Set up event handlers
        self._setup_slack_handlers()
        




    def _setup_slack_handlers(self):
        """Set up Slack event handlers"""
        
        @self.app.event("message")
        async def handle_message_events(event, say, client):
            """Handle message events"""
            await self._handle_message(event, say, client)
        
        @self.app.event("app_mention")
        async def handle_app_mention_events(event, say, client):
            """Handle app mention events"""
            await self._handle_message(event, say, client, is_mention=True)
            
        @self.app.command("/summary")
        async def handle_summary_command(ack, respond, command):
            """Handle /summary slash command"""
            await ack()
            
            if not self.summary_manager:
                await respond("Summary manager not initialized.")
                return
            
            try:
                await self.summary_manager.generate_and_send_daily_summary()
                await respond("Daily summary has been generated and sent to the workspace owner.")
            except Exception as e:
                logging.error(f"Error generating manual summary: {e}")
                await respond(f"Error generating summary: {str(e)}")
        
        @self.app.command("/set-summary-time")
        async def handle_set_summary_time_command(ack, respond, command):
            """Handle /set-summary-time slash command"""
            await ack()
            
            if not self.summary_manager:
                await respond("Summary manager not initialized.")
                return
            
            try:
                # Parse time from command text (format: "HH:MM")
                time_str = command['text'].strip()
                if not time_str:
                    await respond("Please provide time in HH:MM format (e.g., `/set-summary-time 09:00`)")
                    return
                
                hour, minute = map(int, time_str.split(':'))
                if not (0 <= hour <= 23) or not (0 <= minute <= 59):
                    await respond("Invalid time format. Hour must be 0-23, minute must be 0-59.")
                    return
                
                self.summary_manager.summary_time = time(hour, minute)
                await respond(f"Summary time set to {hour:02d}:{minute:02d}")
                
            except ValueError:
                await respond("Invalid time format. Please use HH:MM format (e.g., `09:00`).")
            except Exception as e:
                logging.error(f"Error setting summary time: {e}")
                await respond(f"Error setting summary time: {str(e)}")
        
        @self.app.command("/bot-status")
        async def handle_bot_status_command(ack, respond, command):
            """Handle /bot-status slash command"""
            await ack()
            
            status_text = f"""
*Bot Status:*
• Bot User ID: {self.bot_user_id}
• LLM Provider: {self.llm_client.preferred_llm}
• Messages Processed: {len(self.recent_messages)}
• Respond to Mentions: {self.respond_to_mentions}
• Respond to DMs: {self.respond_to_dms}
• Respond in Threads: {self.respond_in_threads}
"""
            
            if self.summary_manager:
                status_text += f"""
*Daily Summary:*
• Workspace Owner: {self.summary_manager.workspace_owner_id}
• Summary Time: {self.summary_manager.summary_time}
• Last Summary: {self.summary_manager.last_summary_date or 'Never'}
• Next Summary Due: {await self.summary_manager.should_generate_summary()}
"""
            
            await respond(status_text)





    async def _handle_message(self, event, say, client, is_mention=False):
        """Handle incoming messages"""
        try:
            # Skip bot messages
            if event.get("bot_id") or event.get("user") == self.bot_user_id:
                return
            
            # Get message details
            channel = event["channel"]
            user = event.get("user", "unknown")
            text = event.get("text", "")
            ts = event.get("ts", "")
            thread_ts = event.get("thread_ts")
            
            # Store message
            slack_message = SlackMessage(
                channel=channel,
                user=user,
                text=text,
                timestamp=ts,
                thread_ts=thread_ts
            )
            self.recent_messages.append(slack_message)
            
            # Keep only recent messages
            if len(self.recent_messages) > self.max_messages:
                self.recent_messages = self.recent_messages[-self.max_messages:]
            
            # Determine if we should respond
            should_respond = False
            
            # Check if it's a DM
            channel_info = await client.conversations_info(channel=channel)
            is_dm = channel_info["channel"]["is_im"]
            
            if is_dm and self.respond_to_dms:
                should_respond = True
            elif is_mention and self.respond_to_mentions:
                should_respond = True
            elif thread_ts and self.respond_in_threads:
                # Check if we're in a thread where the bot was mentioned or responded
                should_respond = await self._should_respond_in_thread(client, channel, thread_ts)
            
            if should_respond:
                await self._generate_and_send_response(event, say, client, is_dm, is_mention)
                
        except Exception as e:
            logging.error(f"Error handling message: {e}")
            if say:
                await say(f"Sorry, I encountered an error processing your message: {str(e)}")





    async def _should_respond_in_thread(self, client, channel, thread_ts):
        """Check if bot should respond in this thread"""
        try:
            # Get thread history
            result = await client.conversations_replies(
                channel=channel,
                ts=thread_ts,
                limit=10
            )
            
            # Check if bot was mentioned or has participated in this thread
            for message in result.get("messages", []):
                if message.get("user") == self.bot_user_id:
                    return True
                if f"<@{self.bot_user_id}>" in message.get("text", ""):
                    return True
            
            return False
        except Exception as e:
            logging.error(f"Error checking thread participation: {e}")
            return False





    async def _generate_and_send_response(self, event, say, client, is_dm, is_mention):
        """Generate and send LLM response"""
        try:
            user = event.get("user", "unknown")
            text = event.get("text", "")
            channel = event["channel"]
            thread_ts = event.get("thread_ts")
            
            # Get user info for context
            try:
                user_info = await client.users_info(user=user)
                user_name = user_info["user"]["real_name"] or user_info["user"]["name"]
            except Exception as e:
                logging.error(f"Error getting user info: {e}")
                user_name = "User"
            
            # Clean up mention from text if present
            if is_mention:
                text = re.sub(f"<@{self.bot_user_id}>", "", text).strip()
            
            # Get recent context from the channel
            context = await self._get_channel_context(client, channel, exclude_ts=event.get("ts"))
            
            # Generate response
            response = await self.llm_client.get_response(text, context, user_name)
            
            # Send response
            await say(
                text=response,
                thread_ts=thread_ts if thread_ts else None
            )
            
            logging.info(f"Responded to {user_name} in channel {channel}")
            
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            await say("Sorry, I couldn't generate a response right now. Please try again later.")





    async def _get_channel_context(self, client, channel, exclude_ts=None, limit=5):
        """Get recent channel context for LLM"""
        try:
            result = await client.conversations_history(
                channel=channel,
                limit=limit + 1  # Get one extra in case we need to exclude current message
            )
            
            context_messages = []
            for msg in result.get("messages", []):
                # Skip the current message and bot messages
                if (msg.get("ts") == exclude_ts or 
                    msg.get("bot_id") or 
                    msg.get("user") == self.bot_user_id):
                    continue
                
                if msg.get("text") and msg.get("user"):
                    try:
                        user_info = await client.users_info(user=msg["user"])
                        user_name = user_info["user"]["real_name"] or user_info["user"]["name"]
                        context_messages.append(f"{user_name}: {msg['text']}")
                    except Exception as e:
                        logging.error(f"Error getting user info: {e}")
                        context_messages.append(f"User: {msg['text']}")
                
                if len(context_messages) >= limit:
                    break
            
            return "\n".join(reversed(context_messages[-limit:]))
            
        except Exception as e:
            logging.error(f"Error getting channel context: {e}")
            return ""





    async def run_summary_scheduler(self):
        """Run the daily summary scheduler"""
        while True:
            try:
                if self.summary_manager and await self.summary_manager.should_generate_summary():
                    logging.info("Triggering daily summary generation...")
                    await self.summary_manager.generate_and_send_daily_summary()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logging.error(f"Error in summary scheduler: {e}")
                await asyncio.sleep(300)  # Continue checking even if there's an error





    async def start(self):
        """Start the MCP server and Slack bot"""
        try:
            logging.info("Setting up Slack app...")
            await self.setup_slack_app()
            
            logging.info("Starting MCP server...")
            # Start the summary scheduler in the background
            asyncio.create_task(self.run_summary_scheduler())
            
            # Start Slack handler
            logging.info("Starting Slack socket mode handler...")
            await self.handler.start_async()
            
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            raise





    async def stop(self):
        """Stop the server"""
        if self.handler:
            await self.handler.close_async()
        logging.info("Server stopped")
