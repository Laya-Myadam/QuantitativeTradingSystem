"""
Market Data - SIMULATION MODE (No API calls, no rate limits)
Generates realistic market data based on real market patterns
"""
from datetime import datetime, timedelta
from typing import List, Dict
import random
import math

class RealMarketData:
    """
    Simulated market data generator - No API calls
    """
    def __init__(self):
        # Realistic base prices (as of early 2024)
        self.base_prices = {
            'AAPL': 185.00,
            'GOOGL': 140.00,
            'MSFT': 375.00,
            'TSLA': 245.00,
            'AMZN': 150.00,
            'META': 350.00,
            'NVDA': 495.00,
            'AMD': 145.00,
            'NFLX': 480.00,
            'DIS': 90.00,
        }

        # Company information
        self.company_info = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
            'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet Services'},
            'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software'},
            'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive', 'industry': 'Electric Vehicles'},
            'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical', 'industry': 'E-commerce'},
            'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology', 'industry': 'Social Media'},
            'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology', 'industry': 'Semiconductors'},
            'AMD': {'name': 'Advanced Micro Devices', 'sector': 'Technology', 'industry': 'Semiconductors'},
            'NFLX': {'name': 'Netflix Inc.', 'sector': 'Communication Services', 'industry': 'Entertainment'},
            'DIS': {'name': 'The Walt Disney Company', 'sector': 'Communication Services', 'industry': 'Entertainment'},
        }

        # Price state (simulates intraday movement)
        self.current_session = {}
        self.initialize_session()

    def initialize_session(self):
        """Initialize trading session with realistic prices"""
        for symbol, base_price in self.base_prices.items():
            # Simulate overnight gap
            gap = random.gauss(0, 0.02)  # 2% std dev
            open_price = base_price * (1 + gap)

            # Simulate intraday movement
            intraday_move = random.gauss(0, 0.015)  # 1.5% intraday volatility
            current_price = open_price * (1 + intraday_move)

            high = max(open_price, current_price) * random.uniform(1.00, 1.015)
            low = min(open_price, current_price) * random.uniform(0.985, 1.00)

            self.current_session[symbol] = {
                'open': open_price,
                'current': current_price,
                'high': high,
                'low': low,
                'prev_close': base_price,
                'volume': random.randint(20_000_000, 100_000_000)
            }

    def get_current_price(self, symbol: str) -> Dict:
        """Get realistic simulated current price"""
        if symbol not in self.base_prices:
            return {
                'symbol': symbol,
                'price': 0,
                'error': f'Symbol {symbol} not supported',
                'timestamp': datetime.now().isoformat()
            }

        session = self.current_session[symbol]
        info = self.company_info[symbol]

        # Add small random walk
        session['current'] *= random.uniform(0.998, 1.002)
        session['high'] = max(session['high'], session['current'])
        session['low'] = min(session['low'], session['current'])

        change = session['current'] - session['prev_close']
        change_percent = (change / session['prev_close']) * 100

        return {
            'symbol': symbol,
            'price': round(session['current'], 2),
            'open': round(session['open'], 2),
            'high': round(session['high'], 2),
            'low': round(session['low'], 2),
            'volume': session['volume'],
            'change': round(change, 2),
            'change_percent': round(change_percent, 2),
            'timestamp': datetime.now().isoformat(),
            'company_name': info['name'],
            'market_cap': random.randint(500, 3000) * 1e9,
            'pe_ratio': round(random.uniform(15, 45), 2),
            'data_source': 'realistic_simulation',
            'cached': False
        }

    def get_historical_data(self, symbol: str, days: int = 60, interval: str = '1d') -> List[Dict]:
        """Generate realistic historical data"""
        if symbol not in self.base_prices:
            return []

        base_price = self.base_prices[symbol]
        data = []
        current_price = base_price * random.uniform(0.9, 1.0)  # Start slightly lower

        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)

            # Simulate daily returns with realistic parameters
            daily_return = random.gauss(0.001, 0.02)  # 0.1% drift, 2% volatility
            current_price *= (1 + daily_return)

            # Generate OHLC
            open_price = current_price * random.uniform(0.99, 1.01)
            close_price = current_price
            high = max(open_price, close_price) * random.uniform(1.00, 1.02)
            low = min(open_price, close_price) * random.uniform(0.98, 1.00)

            data.append({
                'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp': int(date.timestamp()),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': random.randint(20_000_000, 80_000_000),
                'symbol': symbol
            })

        return data

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            quotes[symbol] = self.get_current_price(symbol)
        return quotes

    def get_market_movers(self) -> Dict:
        """Get simulated market movers"""
        all_symbols = list(self.base_prices.keys())
        movers = []

        for symbol in all_symbols:
            data = self.get_current_price(symbol)
            if data.get('price', 0) > 0:
                movers.append({
                    'symbol': symbol,
                    'name': data['company_name'],
                    'price': data['price'],
                    'change': data['change'],
                    'change_percent': data['change_percent'],
                    'volume': data['volume']
                })

        # Sort
        gainers = sorted([m for m in movers if m['change_percent'] > 0],
                        key=lambda x: x['change_percent'], reverse=True)[:5]
        losers = sorted([m for m in movers if m['change_percent'] < 0],
                       key=lambda x: x['change_percent'])[:5]
        most_active = sorted(movers, key=lambda x: x['volume'], reverse=True)[:5]

        return {
            'gainers': gainers,
            'losers': losers,
            'most_active': most_active,
            'timestamp': datetime.now().isoformat(),
            'note': 'Realistic simulation - no API calls'
        }

    def get_company_info(self, symbol: str) -> Dict:
        """Get company information"""
        if symbol not in self.company_info:
            return {
                'symbol': symbol,
                'name': symbol,
                'sector': 'N/A',
                'industry': 'N/A'
            }

        info = self.company_info[symbol].copy()
        info['symbol'] = symbol

        # Add calculated fields from historical data
        hist = self.get_historical_data(symbol, days=365)
        if hist:
            prices = [d['close'] for d in hist]
            info['52_week_high'] = round(max(prices), 2)
            info['52_week_low'] = round(min(prices), 2)
            info['avg_volume'] = sum(d['volume'] for d in hist) // len(hist)

        info['market_cap'] = random.randint(500, 3000) * 1e9
        info['pe_ratio'] = round(random.uniform(15, 45), 2)
        info['dividend_yield'] = round(random.uniform(0, 3), 2)

        return info

    def search_symbols(self, query: str) -> List[Dict]:
        """Search for symbols"""
        query_lower = query.lower()
        results = []

        for symbol, info in self.company_info.items():
            if (query_lower in symbol.lower() or
                query_lower in info['name'].lower()):
                results.append({
                    'symbol': symbol,
                    'name': info['name'],
                    'type': 'Stock'
                })

        return results

    def clear_cache(self):
        """Reset the session"""
        self.initialize_session()
        print("✓ Session reset")


# Global instance
market_data = RealMarketData()


def get_real_prices(symbols: List[str]) -> Dict:
    """Get prices for multiple symbols"""
    return market_data.get_multiple_quotes(symbols)


def get_historical_prices(symbol: str, days: int = 60) -> List[float]:
    """Get historical closing prices"""
    data = market_data.get_historical_data(symbol, days=days)
    return [d['close'] for d in data] if data else []


def validate_symbol(symbol: str) -> bool:
    """Check if symbol is valid"""
    return symbol in market_data.base_prices