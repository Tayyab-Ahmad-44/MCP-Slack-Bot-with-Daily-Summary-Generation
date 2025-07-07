
import logging
from typing import List, Dict
from datetime import datetime, timedelta, time

from slack_bolt.async_app import AsyncApp

from services.llm_client import LLMClient
from models.data_models import ChannelSummary


class DailySummaryManager:
    """Manages daily summary generation and delivery"""
    
    def __init__(self, app: AsyncApp, llm_client: LLMClient):
        self.app = app
        self.llm_client = llm_client
        self.workspace_owner_id = None
        self.summary_time = time(0, 0)  # 12:00 AM
        self.last_summary_date = None





    async def initialize(self):
        """Initialize the summary manager"""
        await self._get_workspace_owner()





    async def _get_workspace_owner(self):
        """Get the workspace owner ID"""
        try:
            # Get team info to find the primary owner
            team_info = await self.app.client.team_info()
            
            
            print("Team Info: ", team_info)
            
            # Method 1: Try to get primary owner from team info
            if 'team' in team_info and 'primary_owner' in team_info['team']:
                self.workspace_owner_id = team_info['team']['primary_owner']['user_id']
                logging.info(f"Found primary owner: {self.workspace_owner_id}")
                return
                
            # Method 2: Get users list and find admins/owners
            users_result = await self.app.client.users_list()
            for user in users_result['members']:
                if user.get('is_primary_owner', False):
                    self.workspace_owner_id = user['id']
                    logging.info(f"Found primary owner from users list: {self.workspace_owner_id}")
                    return
                elif user.get('is_owner', False) and not self.workspace_owner_id:
                    self.workspace_owner_id = user['id']  # Fallback to any owner
                    
            if not self.workspace_owner_id:
                logging.warning("Could not find workspace owner. Summaries will not be sent.")
                
        except Exception as e:
            logging.error(f"Error getting workspace owner: {e}")





    async def should_generate_summary(self) -> bool:
        """Check if it's time to generate daily summary"""
        now = datetime.now()
        current_date = now.date()
        current_time = now.time()
        
        # Check if it's past summary time and we haven't sent today's summary
        if (current_time >= self.summary_time and 
            (self.last_summary_date is None or self.last_summary_date < current_date)):
            return True
        return False
        




    async def generate_and_send_daily_summary(self):
        """Generate and send daily summary to workspace owner"""
        if not self.workspace_owner_id:
            logging.warning("No workspace owner configured. Skipping daily summary.")
            return
            
        try:
            logging.info("Starting daily summary generation...")
            
            # Get yesterday's date for summary
            yesterday = datetime.now() - timedelta(days=1)
            # yesterday = datetime.now()
            yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Get all channels (public and private)
            channels = await self._get_all_channels()
            
            print(channels)
            
            summaries = []
            
            for channel in channels:
                try:
                    # Get messages from yesterday
                    messages = await self._get_channel_messages_for_date(
                        channel['id'], 
                        yesterday_start, 
                        yesterday_end
                    )
                    
                    print(f"\n\nChannel: {channel['id']},\n Messages: {messages}\n\n")
                    
                    if messages:  # Only summarize channels with activity
                        print("aaa")
                        summary = await self.llm_client.generate_channel_summary(
                            messages, 
                            channel['name']
                        )
                        print("bbb")
                        summary.is_private = channel.get('is_private', False)
                        summaries.append(summary)
                        
                except Exception as e:
                    print("ccc")
                    logging.error(f"Error processing channel {e}")
                    continue
                    
            # Generate and send the comprehensive summary
            if summaries:
                print("Sending Summaries")
                await self._send_summary_to_owner(summaries, yesterday.date())
            else:
                print("Sending No Activity")
                await self._send_no_activity_message(yesterday.date())
                
            self.last_summary_date = datetime.now().date()
            logging.info("Daily summary generation completed")
            
        except Exception as e:
            logging.error(f"Error generating daily summary: {e}")





    async def _get_all_channels(self) -> List[Dict]:
        """Get all channels (public and private) that the bot has access to"""
        channels = []
        
        try:
            # Get public channels
            public_result = await self.app.client.conversations_list(
                types="public_channel",
                exclude_archived=True
            )
            # Filter only channels where bot is a member
            for channel in public_result['channels']:
                if channel.get('is_member', False):
                    channels.append(channel)
            
            # Get private channels (groups)
            private_result = await self.app.client.conversations_list(
                types="private_channel",
                exclude_archived=True
            )
            # Filter only channels where bot is a member
            for channel in private_result['channels']:
                if channel.get('is_member', False):
                    channels.append(channel)
            
            # Get direct messages - bot always has access to DMs
            im_result = await self.app.client.conversations_list(
                types="im",
                exclude_archived=True
            )
            channels.extend(im_result['channels'])
            
            # Get group direct messages - bot always has access to group DMs it's part of
            mpim_result = await self.app.client.conversations_list(
                types="mpim",
                exclude_archived=True
            )
            channels.extend(mpim_result['channels'])
            
        except Exception as e:
            logging.error(f"Error fetching channels: {e}")
            
        return channels

        




    async def _get_channel_messages_for_date(self, channel_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get messages from a specific channel for a specific date range"""
        try:
            # First check if bot has access to this channel
            try:
                channel_info = await self.app.client.conversations_info(channel=channel_id)
                # Check if it's a channel type we can access
                if channel_info['channel'].get('is_channel') and not channel_info['channel'].get('is_member', False):
                    logging.info(f"Skipping channel {channel_id} - bot is not a member")
                    return []
            except Exception as access_error:
                logging.warning(f"Cannot access channel info for {channel_id}: {access_error}")
                return []
            
            result = await self.app.client.conversations_history(
                channel=channel_id,
                oldest=str(start_time.timestamp()),
                latest=str(end_time.timestamp()),
                limit=1000
            )
            return result.get('messages', [])
        except Exception as e:
            # Check if it's the specific "not_in_channel" error
            if "not_in_channel" in str(e):
                logging.info(f"Bot is not a member of channel {channel_id}, skipping...")
                return []
            else:
                logging.error(f"Error fetching messages for channel {channel_id}: {e}")
                return []






    async def _get_channel_messages_for_date(self, channel_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Get messages from a specific channel for a specific date range"""
        try:
            result = await self.app.client.conversations_history(
                channel=channel_id,
                oldest=str(start_time.timestamp()),
                latest=str(end_time.timestamp()),
                limit=1000
            )
            return result.get('messages', [])
        except Exception as e:
            logging.error(f"Error fetching messages for channell {channel_id}: {e}")
            return []
            




    async def _send_summary_to_owner(self, summaries: List[ChannelSummary], date: datetime.date):
        """Send comprehensive summary to workspace owner"""
        try:
            # Create summary message
            summary_text = f"📊 *Daily Workspace Summary for {date.strftime('%B %d, %Y')}*\n\n"
            
            # Separate public and private channels
            public_summaries = [s for s in summaries if not s.is_private]
            private_summaries = [s for s in summaries if s.is_private]
            
            # Add public channels summary
            if public_summaries:
                summary_text += "🌐 *Public Channels:*\n"
                for summary in public_summaries:
                    summary_text += f"\n*#{summary.channel_name}* ({summary.message_count} messages)\n"
                    summary_text += f"_{summary.summary}_\n"
                    if summary.key_topics:
                        summary_text += f"Key topics: {', '.join(summary.key_topics)}\n"
                    summary_text += f"Active users: {len(summary.active_users)}\n"
                    
            # Add private channels summary
            if private_summaries:
                summary_text += "\n🔒 *Private Channels:*\n"
                for summary in private_summaries:
                    summary_text += f"\n*{summary.channel_name}* ({summary.message_count} messages)\n"
                    summary_text += f"_{summary.summary}_\n"
                    if summary.key_topics:
                        summary_text += f"Key topics: {', '.join(summary.key_topics)}\n"
                    summary_text += f"Active users: {len(summary.active_users)}\n"
                    
            # Add overall statistics
            total_messages = sum(s.message_count for s in summaries)
            total_active_users = len(set().union(*[s.active_users for s in summaries]))
            
            summary_text += "\n📈 *Overall Statistics:*\n"
            summary_text += f"• Total messages: {total_messages}\n"
            summary_text += f"• Active channels: {len(summaries)}\n"
            summary_text += f"• Active users: {total_active_users}\n"
            
            # Send DM to workspace owner
            dm_channel = await self.app.client.conversations_open(users=self.workspace_owner_id)
            await self.app.client.chat_postMessage(
                channel=dm_channel['channel']['id'],
                text=summary_text
            )
            
            logging.info(f"Daily summary sent to workspace owner: {self.workspace_owner_id}")
            
        except Exception as e:
            logging.error(f"Error sending summary to owner: {e}")
            




    async def _send_no_activity_message(self, date: datetime.date):
        """Send message when there's no activity to summarize"""
        try:
            message = f"📊 *Daily Workspace Summary for {date.strftime('%B %d, %Y')}*\n\n"
            message += "No significant activity detected in any channels yesterday."
            
            dm_channel = await self.app.client.conversations_open(users=self.workspace_owner_id)
            await self.app.client.chat_postMessage(
                channel=dm_channel['channel']['id'],
                text=message
            )
            
        except Exception as e:
            logging.error(f"Error sending no activity message: {e}")
