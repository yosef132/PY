"""
Trailing Stop Manager
Monitors open positions and moves stop-loss to lock in profits.

How it works:
1. Trade opens normally with fixed SL and TP
2. When price moves 1R in our favor (profit = risk amount), trailing activates
3. SL moves to breakeven (entry price)
4. As price continues, SL trails behind by 0.5x ATR
5. If price reverses, we exit at the trailed SL (still in profit)

This turns losing trades into breakeven and winning trades into bigger wins.
"""

import MetaTrader5 as mt5
from datetime import datetime
from src.utils.config import get_settings
from src.utils.logger import setup_logger

logger = setup_logger("trailing_stop")


class TrailingStopManager:
    """Manages trailing stops for all open positions."""

    def __init__(self):
        settings = get_settings()
        self.symbol = settings.get("symbol", "XAUUSD")
        self.magic = settings.get("execution", {}).get("magic_number", 123456)

        # Trailing config
        self.activation_r = 1.0       # Activate after 1R profit
        self.trail_atr_mult = 0.5     # Trail at 0.5x ATR behind price
        self.breakeven_buffer = 1.0   # Move SL to entry + 1 point (cover spread)
        self.min_trail_distance = 3.0 # Minimum trail distance in price points

        # Track which positions have trailing activated
        self.trailing_active = {}  # ticket -> {"activated": bool, "last_sl": float}

    def update_trailing_stops(self, current_atr: float = None):
        """
        Check all open positions and update trailing stops.
        Call this every scan cycle.

        Args:
            current_atr: Current ATR value for trail distance calculation
        """
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return

        for pos in positions:
            if pos.magic != self.magic:
                continue

            ticket = pos.ticket
            direction = "BUY" if pos.type == 0 else "SELL"
            entry = pos.price_open
            current_sl = pos.sl
            current_tp = pos.tp

            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                continue

            current_price = tick.bid if direction == "BUY" else tick.ask

            # Calculate risk (distance from entry to original SL)
            risk_distance = abs(entry - current_sl)
            if risk_distance <= 0:
                continue

            # Calculate current profit in price distance
            if direction == "BUY":
                profit_distance = current_price - entry
            else:
                profit_distance = entry - current_price

            # Calculate R-multiple
            r_multiple = profit_distance / risk_distance if risk_distance > 0 else 0

            # Initialize tracking
            if ticket not in self.trailing_active:
                self.trailing_active[ticket] = {
                    "activated": False,
                    "original_sl": current_sl,
                    "best_price": entry,
                }

            state = self.trailing_active[ticket]

            # Update best price seen
            if direction == "BUY":
                state["best_price"] = max(state["best_price"], current_price)
            else:
                if state["best_price"] == entry:
                    state["best_price"] = current_price
                else:
                    state["best_price"] = min(state["best_price"], current_price)

            # Step 1: Activate trailing when profit reaches 1R
            if not state["activated"] and r_multiple >= self.activation_r:
                # Move SL to breakeven
                if direction == "BUY":
                    new_sl = entry + self.breakeven_buffer
                else:
                    new_sl = entry - self.breakeven_buffer

                if self._modify_sl(ticket, new_sl, pos.volume):
                    state["activated"] = True
                    logger.info(f"  Trailing ACTIVATED #{ticket} | Moved SL to breakeven: {new_sl:.2f}")
                continue

            # Step 2: Trail the stop as price moves further
            if state["activated"]:
                # Calculate trail distance
                trail_distance = max(
                    current_atr * self.trail_atr_mult if current_atr else risk_distance * 0.5,
                    self.min_trail_distance
                )

                if direction == "BUY":
                    new_sl = state["best_price"] - trail_distance
                    # Only move SL up, never down
                    if new_sl > current_sl + 0.5:
                        if self._modify_sl(ticket, new_sl, pos.volume):
                            logger.info(f"  Trailing #{ticket} | SL: {current_sl:.2f} -> {new_sl:.2f} "
                                        f"(price: {current_price:.2f}, best: {state['best_price']:.2f})")
                else:
                    new_sl = state["best_price"] + trail_distance
                    # Only move SL down, never up
                    if new_sl < current_sl - 0.5:
                        if self._modify_sl(ticket, new_sl, pos.volume):
                            logger.info(f"  Trailing #{ticket} | SL: {current_sl:.2f} -> {new_sl:.2f} "
                                        f"(price: {current_price:.2f}, best: {state['best_price']:.2f})")

    def _modify_sl(self, ticket: int, new_sl: float, volume: float) -> bool:
        """Send order to modify SL on MT5."""
        try:
            # Get current position to read TP
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False

            pos = positions[0]
            symbol_info = mt5.symbol_info(self.symbol)
            digits = symbol_info.digits if symbol_info else 2
            new_sl = round(new_sl, digits)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": pos.tp,  # Keep original TP
            }

            result = mt5.order_send(request)

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
            else:
                error = result.comment if result else "Unknown"
                logger.warning(f"  Failed to modify SL #{ticket}: {error}")
                return False

        except Exception as e:
            logger.warning(f"  SL modify error #{ticket}: {e}")
            return False

    def cleanup_closed(self):
        """Remove tracking for positions that are no longer open."""
        open_tickets = set()
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                if pos.magic == self.magic:
                    open_tickets.add(pos.ticket)

        # Remove closed positions from tracking
        closed = [t for t in self.trailing_active if t not in open_tickets]
        for t in closed:
            del self.trailing_active[t]

    def get_status(self) -> list:
        """Get status of all tracked positions."""
        status = []
        for ticket, state in self.trailing_active.items():
            status.append({
                "ticket": ticket,
                "trailing_active": state["activated"],
                "original_sl": state["original_sl"],
                "best_price": state["best_price"],
            })
        return status
