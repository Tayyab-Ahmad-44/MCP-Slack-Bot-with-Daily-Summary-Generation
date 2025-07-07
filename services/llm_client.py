

import os
import json
import httpx
import logging
from typing import List, Dict

from models.data_models import ChannelSummary

class LLMClient:
    """Handles LLM API calls for Claude, OpenAI, and Groq"""
    
    def __init__(self, preferred_llm):
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.preferred_llm = preferred_llm

        
    async def get_response(self, message: str, context: str = "", user_name: str = "User") -> str:
        """Get response from preferred LLM"""
        try:
            if self.preferred_llm == "claude" and self.claude_api_key:
                return await self._call_claude(message, context, user_name)
            elif self.preferred_llm == "openai" and self.openai_api_key:
                return await self._call_openai(message, context, user_name)
            elif self.preferred_llm == "groq" and self.groq_api_key:
                return await self._call_groq(message, context, user_name)
            else:
                return "Sorry, I don't have access to an LLM. Please configure ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in your .env file."
        except Exception as e:
            logging.error(f"LLM API error: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"

    async def generate_channel_summary(self, messages: List[Dict], channel_name: str) -> ChannelSummary:
        """Generate a summary for a channel based on messages"""
        if not messages:
            return ChannelSummary(
                channel_id="",
                channel_name=channel_name,
                message_count=0,
                summary="No activity in this channel today.",
                key_topics=[],
                active_users=[],
                is_private=False
            )

        # Prepare message text for analysis
        message_texts = []
        users = set()
        for msg in messages:
            if msg.get('text') and not msg.get('bot_id'):  # Skip bot messages
                message_texts.append(f"{msg.get('user', 'Unknown')}: {msg['text']}")
                if msg.get('user'):
                    users.add(msg['user'])

        conversation_text = "\n".join(message_texts[-50:])  # Last 50 messages
        
        summary_prompt = f"""
        Analyze the following Slack channel conversation from #{channel_name} and provide a structured summary.

        Conversation:
        {conversation_text}

        Please provide:
        1. A brief summary of the main discussions (2-3 sentences)
        2. Key topics discussed (up to 5 topics)
        3. Notable decisions or action items (if any)

        Format your response as JSON with keys: summary, key_topics, action_items
        """

        try:
            response = await self.get_response(summary_prompt)
            
            # Try to parse JSON response, fallback to basic summary if parsing fails
            try:
                summary_data = json.loads(response)
                summary_text = summary_data.get('summary', response)
                key_topics = summary_data.get('key_topics', [])
                action_items = summary_data.get('action_items', [])
                
                # Add action items to summary if present
                if action_items:
                    summary_text += f"\n\nAction Items: {', '.join(action_items)}"
                    
            except json.JSONDecodeError:
                summary_text = response
                key_topics = []

            return ChannelSummary(
                channel_id=messages[0].get('channel', ''),
                channel_name=channel_name,
                message_count=len(messages),
                summary=summary_text,
                key_topics=key_topics,
                active_users=list(users),
                is_private=False  # Will be set by caller
            )
            
        except Exception as e:
            logging.error(f"Error generating summary for {channel_name}: {e}")
            return ChannelSummary(
                channel_id=messages[0].get('channel', ''),
                channel_name=channel_name,
                message_count=len(messages),
                summary=f"Summary generation failed: {str(e)}",
                key_topics=[],
                active_users=list(users),
                is_private=False
            )

    async def _call_claude(self, message: str, context: str, user_name: str) -> str:
        """Call Claude API"""
        async with httpx.AsyncClient() as client:
            headers = {
                "x-api-key": self.claude_api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            system_prompt = f"""You are a helpful Slack bot assistant. You're responding to messages in a Slack workspace.

Current user: {user_name}
Recent conversation context:
{context}

Guidelines:
- Be helpful, friendly, and concise
- Use Slack-appropriate formatting (like *bold* or `code`)
- Keep responses under 2000 characters
- If asked about your capabilities, mention you can help with questions, analysis, coding, and general assistance
- You can see recent channel history for context"""

            payload = {
                "model": "claude-3-haiku-20240307",  
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
                ],
                "system": system_prompt
            }
            
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                logging.error(f"Claude API error: {response.status_code} - {response.text}")
                return f"Sorry, I couldn't process your request right now. (Claude API error: {response.status_code})"

    async def _call_openai(self, message: str, context: str, user_name: str) -> str:
        """Call OpenAI API"""
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            system_message = f"""You are a helpful Slack bot assistant responding to messages in a Slack workspace.

Current user: {user_name}
Recent conversation context:
{context}

Guidelines:
- Be helpful, friendly, and concise
- Use Slack-appropriate formatting
- Keep responses under 2000 characters
- You can see recent channel history for context"""

            payload = {
                "model": "gpt-4-turbo-preview",  # Fixed model name
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logging.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return f"Sorry, I couldn't process your request right now. (OpenAI API error: {response.status_code})"

    async def _call_groq(self, message: str, context: str, user_name: str) -> str:
        """Call Groq API"""
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            system_message = f"""You are a helpful Slack bot assistant responding to messages in a Slack workspace.

Current user: {user_name}
Recent conversation context:
{context}

Guidelines:
- Be helpful, friendly, and concise
- Use Slack-appropriate formatting (like *bold* or `code`)
- Keep responses under 2000 characters
- If asked about your capabilities, mention you can help with questions, analysis, coding, and general assistance
- You can see recent channel history for context"""

            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }

            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logging.error(f"Groq API error: {response.status_code} - {response.text}")
                return f"Sorry, I couldn't process your request right now. (Groq API error: {response.status_code})"
