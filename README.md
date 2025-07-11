# MCP-Slack-Bot-with-Daily-Summary-Generation

A powerful Slack bot that integrates with Model Context Protocol (MCP) and provides LLM-powered responses using Claude, OpenAI, or Groq APIs. Features include automatic daily summaries, message handling, and comprehensive workspace monitoring.

## Features

- 🤖 **LLM Integration**: Supports Claude, OpenAI, and Groq APIs
- 📊 **Daily Summaries**: Automatic workspace activity summaries
- 💬 **Smart Responses**: Responds to mentions, DMs, and thread conversations
- 🔧 **MCP Compatible**: Works as both standalone bot and MCP server
- ⚙️ **Configurable**: Flexible settings for response behavior
- 📈 **Analytics**: Message tracking and workspace insights

## Prerequisites

- Python 3.8 or higher
- Slack workspace with admin privileges
- At least one LLM API key (Claude, OpenAI, or Groq)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd slack-mcp-server
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Create environment file**
   ```bash
   cp .env.example .env
   ```

## Slack App Setup

### Step 1: Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **"Create New App"**
3. Choose **"From scratch"**
4. Enter app name (e.g., "MCP Bot") and select your workspace
5. Click **"Create App"**

### Step 2: Configure OAuth & Permissions

1. In your app settings, go to **"OAuth & Permissions"**
2. Scroll down to **"Scopes"** section
3. Add the following **Bot Token Scopes**:
   ```
   app_mentions:read
   channels:history
   channels:read
   chat:write
   conversations.connect
   groups:history
   groups:read
   im:history
   im:read
   im:write
   mpim:history
   mpim:read
   reactions:write
   team:read
   users:read
   ```

4. Scroll up and click **"Install to Workspace"**
5. Authorize the app
6. Copy the **"Bot User OAuth Token"** (starts with `xoxb-`)

### Step 3: Enable Socket Mode

1. Go to **"Socket Mode"** in your app settings
2. Toggle **"Enable Socket Mode"** to ON
3. Create a new token:
   - Token Name: `socket-token`
   - Add scope: `connections:write`
4. Copy the **"App-Level Token"** (starts with `xapp-`)

### Step 4: Configure Event Subscriptions

1. Go to **"Event Subscriptions"**
2. Toggle **"Enable Events"** to ON
3. In **"Subscribe to bot events"**, add:
   ```
   app_mention
   message.channels
   message.groups
   message.im
   message.mpim
   ```


## Environment Configuration

Edit your `.env` file with the following variables:

```bash
# Slack Configuration (Required)
SLACK_BOT_TOKEN=xoxb-your-bot-token-here
SLACK_APP_TOKEN=xapp-your-app-token-here

# LLM API Keys (At least one required)
ANTHROPIC_API_KEY=your-claude-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
GROQ_API_KEY=your-groq-api-key-here
```

### Getting LLM API Keys

**Claude (Anthropic):**
1. Go to [https://console.anthropic.com](https://console.anthropic.com)
2. Create account and get API key from dashboard

**OpenAI:**
1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create account and generate API key

**Groq:**
1. Go to [https://console.groq.com](https://console.groq.com)
2. Create account and get API key (free tier available)

## Running the Application

Run the bot as a standalone Slack application:

```bash
python main.py
```



## Usage

### Bot Interactions

1. **Direct Messages**: Bot responds to all DMs automatically
2. **Mentions**: Use `@YourBot` in any channel
3. **Threads**: Bot participates in threads where it was mentioned
4. **Reactions**: Bot can add emoji reactions to messages

### Slash Commands

- `/summary` - Generate immediate daily summary
- `/set-summary-time 09:00` - Set daily summary time
- `/bot-status` - Check bot configuration and status

### Daily Summaries

The bot automatically generates daily workspace summaries including:
- Channel activity overview
- Key discussion topics
- Active user statistics
- Private channel summaries (if bot has access)

## MCP Tools Available

When running as MCP server, these tools are available:

1. **send_slack_message** - Send messages to channels
2. **get_channel_history** - Retrieve channel message history
3. **ask_llm** - Query the configured LLM
4. **react_to_message** - Add emoji reactions
5. **generate_manual_summary** - Trigger summary generation
6. **set_summary_time** - Configure summary timing

## Configuration Options

### Bot Behavior

Edit these variables in `SlackMCPServer.__init__()`:

```python
self.respond_to_mentions = True    # Respond to @mentions
self.respond_to_dms = True         # Respond to direct messages
self.respond_in_threads = True     # Participate in threads
self.max_messages = 50             # Message history limit
```

### LLM Selection

Set preferred LLM in `SlackMCPServer.__init__()`:

```python
self.llm_client = LLMClient(preferred_llm='groq')  # Options: claude, openai, groq
```

### Summary Timing

Default summary time is midnight (00:00). Change in `DailySummaryManager.__init__()`:

```python
self.summary_time = time(9, 0)  # 9:00 AM
```

## Dependencies

```
slack-bolt>=1.14.0
python-dotenv>=1.0.0
httpx>=0.24.0
mcp>=0.1.0
asyncio
```

## Bye!
