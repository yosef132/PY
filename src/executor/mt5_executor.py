"""
MT5 Trade Executor
Actually places BUY/SELL orders on MetaTrader 5.
This is the "hands" of the bot — it executes what the brain decides.
"""

import MetaTrader5 as mt5
import time
from datetime import datetime
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("mt5_executor")


class MT5Executor:
    """Places and manages trades on MetaTrader 5."""

    def __init__(self):
        settings = get_settings()
        self.exec_config = settings.get("execution", {})
        self.symbol = settings.get("symbol", "XAUUSD")
        self.magic = self.exec_config.get("magic_number", 123456)
        self.slippage = self.exec_config.get("slippage_points", 30)
        self.is_demo = self.exec_config.get("mode", "demo") == "demo"

    def place_trade(self, direction: str, lot_size: float,
                    stop_loss: float, take_profit: float,
                    comment: str = "") -> dict:
        """
        Place a trade on MT5.

        Args:
            direction: "BUY" or "SELL"
            lot_size: Position size (e.g. 0.01)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Trade comment for identification

        Returns:
            Dict with trade result info
        """
        # Get current price
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("  Cannot get tick data!")
            return {"success": False, "error": "No tick data"}

        # Determine order type and price
        if direction == "BUY":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        elif direction == "SELL":
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            return {"success": False, "error": f"Invalid direction: {direction}"}

        # Validate SL/TP — reject if wrong-sided (stale signal from a historical bar)
        if direction == "BUY":
            if stop_loss >= price:
                logger.warning(f"  REJECTED: BUY SL {stop_loss:.2f} is above price {price:.2f} (stale signal)")
                return {"success": False, "error": "SL wrong side — stale signal"}
            if take_profit <= price:
                logger.warning(f"  REJECTED: BUY TP {take_profit:.2f} is below price {price:.2f} (stale signal)")
                return {"success": False, "error": "TP wrong side — stale signal"}
        else:
            if stop_loss <= price:
                logger.warning(f"  REJECTED: SELL SL {stop_loss:.2f} is below price {price:.2f} (stale signal)")
                return {"success": False, "error": "SL wrong side — stale signal"}
            if take_profit >= price:
                logger.warning(f"  REJECTED: SELL TP {take_profit:.2f} is above price {price:.2f} (stale signal)")
                return {"success": False, "error": "TP wrong side — stale signal"}

        # Round to proper decimals
        symbol_info = mt5.symbol_info(self.symbol)
        digits = symbol_info.digits if symbol_info else 2

        price = round(price, digits)
        stop_loss = round(stop_loss, digits)
        take_profit = round(take_profit, digits)
        lot_size = round(lot_size, 2)

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": comment[:31],  # MT5 limits comment to 31 chars
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(f"  Sending order: {direction} {lot_size} lots @ {price} | "
                     f"SL: {stop_loss} | TP: {take_profit}")

        # Send order
        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            logger.error(f"  Order send failed: {error}")
            return {"success": False, "error": str(error)}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"  Order rejected: {result.retcode} - {result.comment}")
            return {
                "success": False,
                "retcode": result.retcode,
                "comment": result.comment,
                "error": f"Retcode {result.retcode}: {result.comment}"
            }

        # Success!
        logger.info(f"  ORDER FILLED: Ticket #{result.order} | {direction} {lot_size} @ {result.price}")

        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "direction": direction,
            "sl": stop_loss,
            "tp": take_profit,
            "comment": comment,
            "time": datetime.now().isoformat(),
        }

    def get_open_positions(self) -> list:
        """Get all open positions for our symbol with our magic number."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []

        our_positions = []
        for pos in positions:
            if pos.magic == self.magic:
                our_positions.append({
                    "ticket": pos.ticket,
                    "direction": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "time": datetime.fromtimestamp(pos.time).isoformat(),
                    "comment": pos.comment,
                })

        return our_positions

    def close_position(self, ticket: int) -> dict:
        """Close a specific position by ticket number."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = positions[0]
        tick = mt5.symbol_info_tick(self.symbol)

        if pos.type == 0:  # BUY position -> close with SELL
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:  # SELL position -> close with BUY
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.slippage,
            "magic": self.magic,
            "comment": "AI_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"  Position #{ticket} CLOSED at {result.price} | PnL: ${pos.profit:.2f}")
            return {"success": True, "ticket": ticket, "profit": pos.profit}
        else:
            error = result.comment if result else mt5.last_error()
            logger.error(f"  Failed to close #{ticket}: {error}")
            return {"success": False, "error": str(error)}

    def close_all_positions(self) -> list:
        """Close all positions opened by this bot."""
        positions = self.get_open_positions()
        results = []
        for pos in positions:
            result = self.close_position(pos["ticket"])
            results.append(result)
        return results

    def get_account_summary(self) -> dict:
        """Get current account info."""
        info = mt5.account_info()
        if info is None:
            return {}
        return {
            "balance": info.balance,
            "equity": info.equity,
            "profit": info.profit,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level if info.margin_level else 0,
        }
