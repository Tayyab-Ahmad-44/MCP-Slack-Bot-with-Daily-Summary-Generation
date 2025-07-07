
from dataclasses import dataclass
from typing import List, Optional



@dataclass
class ChannelSummary:
    channel_id: str
    channel_name: str
    message_count: int
    summary: str
    key_topics: List[str]
    active_users: List[str]
    is_private: bool



@dataclass
class SlackMessage:
    channel: str
    user: str
    text: str
    timestamp: str
    thread_ts: Optional[str] = None