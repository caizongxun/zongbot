"""Discord Bot for Real-time Trading Signals - Phase 4

Automated Discord bot that sends trading signals and updates market data.
"""

import logging
import asyncio
from typing import Optional, Dict
from datetime import datetime

import discord
from discord.ext import commands, tasks
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ZongBot(commands.Cog):
    """Discord bot for ZongBot trading signals."""
    
    def __init__(self, bot: commands.Bot):
        """Initialize bot cog.
        
        Args:
            bot: Discord bot instance
        """
        self.bot = bot
        self.channel_id = int(os.getenv('DISCORD_CHANNEL_ID', 0))
        self.signal_cooldown = int(os.getenv('SIGNAL_COOLDOWN', 300))  # 5 minutes
        self.last_signal_time = {}
        
        logger.info("ZongBot cog initialized")
    
    @commands.Cog.listener()
    async def on_ready(self):
        """Called when bot connects to Discord."""
        logger.info(f"{self.bot.user} has connected to Discord")
        self.send_status_updates.start()
    
    @commands.command(name='ping')
    async def ping(self, ctx):
        """Test command - responds with pong."""
        await ctx.send(f'Pong! ({self.bot.latency*1000:.2f}ms)')
    
    @commands.command(name='status')
    async def status(self, ctx):
        """Get current bot status."""
        embed = discord.Embed(
            title="ZongBot Status",
            color=discord.Color.green()
        )
        embed.add_field(name="Status", value="🟢 Online", inline=False)
        embed.add_field(name="Uptime", value="N/A", inline=False)
        embed.add_field(name="Version", value="0.1.0", inline=False)
        await ctx.send(embed=embed)
    
    async def send_trading_signal(self, signal: Dict):
        """Send trading signal to Discord channel.
        
        Args:
            signal: Signal dictionary containing prediction and action
        """
        if not self.channel_id:
            logger.warning("Channel ID not configured")
            return
        
        channel = self.bot.get_channel(self.channel_id)
        if not channel:
            logger.warning(f"Channel {self.channel_id} not found")
            return
        
        # Check cooldown
        symbol = signal.get('symbol', 'UNKNOWN')
        last_time = self.last_signal_time.get(symbol, 0)
        if datetime.now().timestamp() - last_time < self.signal_cooldown:
            logger.debug(f"Signal cooldown active for {symbol}")
            return
        
        # Create embed
        embed = self._create_signal_embed(signal)
        
        try:
            await channel.send(embed=embed)
            self.last_signal_time[symbol] = datetime.now().timestamp()
            logger.info(f"Signal sent for {symbol}")
        except Exception as e:
            logger.error(f"Error sending signal: {e}")
    
    def _create_signal_embed(self, signal: Dict) -> discord.Embed:
        """Create Discord embed for signal.
        
        Args:
            signal: Signal dictionary
        
        Returns:
            Discord embed object
        """
        action = signal.get('action', 'HOLD')
        symbol = signal.get('symbol', 'UNKNOWN')
        timeframe = signal.get('timeframe', '1h')
        confidence = signal.get('confidence', 0)
        volatility = signal.get('volatility', 0)
        strength = signal.get('strength', 0)
        
        # Color based on action
        color = discord.Color.green() if action == 'BUY' else discord.Color.red()
        
        embed = discord.Embed(
            title=f"🚨 {action} Signal - {symbol} ({timeframe})",
            color=color,
            timestamp=datetime.now()
        )
        
        embed.add_field(
            name="Action",
            value=f"**{action}**",
            inline=True
        )
        
        embed.add_field(
            name="Confidence",
            value=f"{confidence:.1%}",
            inline=True
        )
        
        embed.add_field(
            name="Volatility",
            value=f"{volatility:.4f}",
            inline=True
        )
        
        embed.add_field(
            name="Signal Strength",
            value=self._format_strength_bar(strength),
            inline=False
        )
        
        embed.set_footer(text="ZongBot Trading System")
        
        return embed
    
    @staticmethod
    def _format_strength_bar(strength: float) -> str:
        """Format signal strength as visual bar.
        
        Args:
            strength: Strength value 0-1
        
        Returns:
            Formatted strength bar string
        """
        bars = int(strength * 10)
        return ''.join(['🟩' if i < bars else '⬜' for i in range(10)])
    
    @tasks.loop(minutes=5)
    async def send_status_updates(self):
        """Periodically send status updates."""
        if not self.channel_id:
            return
        
        channel = self.bot.get_channel(self.channel_id)
        if not channel:
            return
        
        try:
            # Get system status (placeholder)
            embed = discord.Embed(
                title="📊 Market Status Update",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            embed.add_field(
                name="Symbols Monitored",
                value="15 cryptocurrencies",
                inline=True
            )
            
            embed.add_field(
                name="Timeframes",
                value="15m, 1h, 4h",
                inline=True
            )
            
            embed.add_field(
                name="Status",
                value="🟢 Operational",
                inline=True
            )
            
            embed.set_footer(text="ZongBot Trading System")
            
            # Send once per hour
            if datetime.now().minute == 0:
                await channel.send(embed=embed)
        
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
    
    @commands.command(name='help_trading')
    async def help_trading(self, ctx):
        """Display trading help information."""
        embed = discord.Embed(
            title="ZongBot Trading Help",
            description="Real-time cryptocurrency trading signals powered by AI",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="BUY Signal",
            value="Indicates predicted upward price movement with acceptable volatility",
            inline=False
        )
        
        embed.add_field(
            name="SELL Signal",
            value="Indicates predicted downward price movement with acceptable volatility",
            inline=False
        )
        
        embed.add_field(
            name="Confidence",
            value="Model confidence in the prediction (0-100%)",
            inline=False
        )
        
        embed.add_field(
            name="Signal Strength",
            value="Combined metric considering confidence and volatility",
            inline=False
        )
        
        embed.add_field(
            name="Commands",
            value="`.ping` - Test bot connectivity\n`.status` - View bot status\n`.help_trading` - Show this help",
            inline=False
        )
        
        await ctx.send(embed=embed)


async def setup(bot: commands.Bot):
    """Load the cog.
    
    Args:
        bot: Discord bot instance
    """
    await bot.add_cog(ZongBot(bot))


class DiscordBotManager:
    """Manager for Discord bot lifecycle."""
    
    def __init__(self):
        """Initialize bot manager."""
        self.token = os.getenv('DISCORD_TOKEN')
        self.bot = None
        
        if not self.token:
            raise ValueError("DISCORD_TOKEN not set in environment")
    
    async def start(self):
        """Start the Discord bot."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guild_messages = True
        
        self.bot = commands.Bot(command_prefix='!', intents=intents)
        
        @self.bot.event
        async def on_ready():
            logger.info(f"{self.bot.user} is now running!")
        
        # Load cog
        await setup(self.bot)
        
        # Start bot
        await self.bot.start(self.token)
    
    async def stop(self):
        """Stop the Discord bot gracefully."""
        if self.bot and self.bot.is_closed() is False:
            await self.bot.close()
            logger.info("Bot stopped")
    
    async def send_signal(self, signal: Dict):
        """Send trading signal.
        
        Args:
            signal: Signal dictionary
        """
        if self.bot and not self.bot.is_closed():
            cog = self.bot.get_cog('ZongBot')
            if cog:
                await cog.send_trading_signal(signal)


async def main():
    """Example bot usage."""
    manager = DiscordBotManager()
    
    try:
        await manager.start()
    except KeyboardInterrupt:
        await manager.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run bot
    asyncio.run(main())
