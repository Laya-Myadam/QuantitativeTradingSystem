"""
Data Pipeline - Simulated Market Data
Add this in Phase 2
"""
import random
from datetime import datetime, timedelta


def get_market_data(symbol, days=30):
    """
    Get simulated market data
    """
    data = []
    base_price = random.uniform(100, 300)

    for i in range(days):
        date = datetime.now() - timedelta(days=days - i)
        change = random.uniform(-5, 5)
        base_price += change

        data.append({
            "date": date.isoformat(),
            "symbol": symbol,
            "open": round(base_price, 2),
            "high": round(base_price + random.uniform(0, 3), 2),
            "low": round(base_price - random.uniform(0, 3), 2),
            "close": round(base_price, 2),
            "volume": random.randint(1000000, 10000000)
        })

    return data


def get_current_price(symbol):
    """Get current price for symbol"""
    return round(random.uniform(100, 500), 2)