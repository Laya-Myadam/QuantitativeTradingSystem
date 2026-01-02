"""
Main Backend Server - START HERE!
This file runs everything
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import random

# Create the app
app = FastAPI(title="Quantitative Trading System")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (simple storage)
trading_state = {
    "active": False,
    "portfolio_value": 1000000.0,
    "cash": 1000000.0,
    "positions": {},
    "trades": []
}


# ============================================
# BASIC ENDPOINTS
# ============================================

@app.get("/")
def home():
    """Home page - test if server is running"""
    return {
        "message": "🚀 Trading System is Running!",
        "status": "online",
        "docs": "Visit http://localhost:8000/docs"
    }


@app.get("/health")
def health():
    """System health check"""
    return {
        "status": "healthy",
        "trading_active": trading_state["active"],
        "timestamp": datetime.now().isoformat(),
        "uptime": "99.9%"
    }


# ============================================
# TRADING CONTROL
# ============================================

@app.post("/trading/start")
def start_trading():
    """Start automated trading"""
    trading_state["active"] = True
    return {
        "status": "started",
        "message": "✅ Trading activated!",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/trading/stop")
def stop_trading():
    """Stop automated trading"""
    trading_state["active"] = False
    return {
        "status": "stopped",
        "message": "⏸️ Trading stopped!",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# PORTFOLIO & METRICS
# ============================================

@app.get("/portfolio")
def get_portfolio():
    """Get current portfolio state"""
    # Simulate changes when trading
    if trading_state["active"]:
        change = random.uniform(-500, 1000)
        trading_state["portfolio_value"] += change

    pnl = trading_state["portfolio_value"] - 1000000

    return {
        "value": round(trading_state["portfolio_value"], 2),
        "cash": round(trading_state["cash"], 2),
        "positions": trading_state["positions"],
        "pnl": round(pnl, 2),
        "pnl_percent": round((pnl / 1000000) * 100, 2)
    }


@app.get("/metrics")
def get_metrics():
    """Get performance metrics"""
    return {
        "alpha": round(random.uniform(0.5, 1.0), 2),
        "sharpe_ratio": round(random.uniform(1.5, 2.5), 2),
        "max_drawdown": round(random.uniform(-10, -3), 2),
        "win_rate": round(random.uniform(55, 65), 1),
        "total_trades": random.randint(100, 500),
        "volatility": round(random.uniform(10, 20), 1)
    }


# ============================================
# TRADING ACTIVITY
# ============================================

@app.get("/trades")
def get_recent_trades(limit: int = 20):
    """Get recent trading activity"""
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]

    trades = []
    for i in range(limit):
        symbol = random.choice(symbols)
        action = random.choice(["BUY", "SELL"])
        quantity = random.randint(10, 100)
        price = round(random.uniform(100, 500), 2)

        trades.append({
            "id": i,
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "total": round(quantity * price, 2),
            "timestamp": datetime.now().isoformat()
        })

    return {"trades": trades, "count": len(trades)}


# ============================================
# PREDICTIONS (BASIC)
# ============================================

@app.get("/predictions/{symbol}")
def get_predictions(symbol: str, days: int = 10):
    """Get price predictions"""
    base_price = random.uniform(100, 300)
    predictions = []

    for i in range(days):
        change = random.uniform(-5, 5)
        base_price += change
        predictions.append({
            "day": i + 1,
            "price": round(base_price, 2),
            "confidence": round(random.uniform(70, 95), 1)
        })

    return {
        "symbol": symbol,
        "predictions": predictions,
        "model": "Basic Simulation"
    }


# ============================================
# STRATEGIES
# ============================================

@app.get("/strategies")
def get_strategies():
    """List available strategies"""
    return {
        "strategies": [
            {
                "id": "rl_bandit",
                "name": "RL Multi-Armed Bandit",
                "status": "ready",
                "description": "Reinforcement learning portfolio allocation"
            },
            {
                "id": "lstm_forecast",
                "name": "LSTM Time Series",
                "status": "ready",
                "description": "Deep learning price prediction"
            },
            {
                "id": "arima",
                "name": "ARIMA Statistical",
                "status": "ready",
                "description": "Statistical forecasting model"
            }
        ]
    }

# ============================================
# RL AGENT ENDPOINTS
# ============================================

from models.rl_agent import get_rl_recommendation, simulate_trade_and_update, get_agent_stats

@app.get("/rl/recommendation")
def get_rl_trading_recommendation():
    """Get RL agent's trading recommendation"""
    return get_rl_recommendation()

@app.post("/rl/trade")
def execute_rl_trade():
    """Execute trade based on RL recommendation"""
    result = simulate_trade_and_update()
    return result

@app.get("/rl/stats")
def get_rl_statistics():
    """Get RL agent performance statistics"""
    return get_agent_stats()


# ============================================
# FORECASTING ENDPOINTS
# ============================================

from models.forecaster import get_price_forecast, forecaster


@app.get("/forecast/{symbol}")
def forecast_price(symbol: str, days: int = 10):
    """
    Get price forecast using ensemble of LSTM, ARIMA, Prophet
    """
    return get_price_forecast(symbol, days)


@app.get("/forecast/models/{symbol}")
def forecast_all_models(symbol: str, days: int = 10):
    """
    See predictions from each individual model
    """
    from models.forecaster import generate_historical_prices

    historical = generate_historical_prices(symbol, days=60)
    forecast = forecaster.predict(historical, steps=days)

    return {
        'symbol': symbol,
        'current_price': round(historical[-1], 2),
        'ensemble_forecast': forecast['ensemble'],
        'individual_models': forecast['models'],
        'confidence_intervals': forecast['confidence_intervals'],
        'model_weights': forecast['weights']
    }


@app.get("/forecast/accuracy")
def get_forecast_accuracy():
    """
    Get forecasting accuracy metrics
    """
    metrics = forecaster.calculate_accuracy_metrics()
    return {
        'metrics': metrics,
        'description': {
            'rmse': 'Root Mean Squared Error (lower is better)',
            'mae': 'Mean Absolute Error (lower is better)',
            'mape': 'Mean Absolute Percentage Error (lower is better)'
        }
    }

# ============================================
# RUN THE SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("🚀 QUANTITATIVE TRADING SYSTEM - STARTING")
    print("=" * 70)
    print("\n📊 Dashboard: Open 'dashboard.html' in your browser")
    print("📍 API Server: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("❤️  Health: http://localhost:8000/health")
    print("\n" + "=" * 70)
    print("Press CTRL+C to stop the server")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)