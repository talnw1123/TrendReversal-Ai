"""
AI Trading Assistant API — FastAPI version of llm_hybrid_agent.py

Endpoints:
  POST /api/chat          — AI analysis with Mixture-of-Experts (2 experts + 1 master)
  POST /api/chat/simple   — Single-model quick response
  GET  /api/signals       — Today's live trading signals
  GET  /api/history       — SQL history (last N days)
  GET  /api/performance   — Strategy performance stats
  GET  /api/markets       — List available markets
  POST /api/signals/save  — Save today's signals to DB

Run:  uvicorn workflows.api_server:app --reload --port 8000
"""

import os
import sys
import json
import sqlite3
import datetime
import subprocess
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_OLLAMA = False
OLLAMA_MODEL = "llama3.2:1b"
EXTERNAL_MODEL = "anthropic/claude-3.5-sonnet"
EXTERNAL_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
EXTERNAL_BASE_URL = "https://openrouter.ai/api/v1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
DB_FILE = os.path.join(SCRIPT_DIR, "trading_database.sqlite")

# Expert models (MoE)
EXPERT1_MODEL = "deepseek/deepseek-r1-distill-llama-70b"
EXPERT2_MODEL = "qwen/qwen-2.5-72b-instruct"
MASTER_MODEL = "anthropic/claude-3-5-haiku"


# ─── Database helpers (reused from llm_hybrid_agent.py) ───

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS strategy_performance (
            market TEXT PRIMARY KEY,
            base_return_pct REAL,
            bnh_return_pct REAL,
            win_rate_pct REAL,
            total_trades INTEGER,
            max_drawdown_pct REAL,
            updated_at TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_sql_history(days: int = 5) -> str:
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    perf_text = "== ภาพรวมประสิทธิภาพระบบเทรด 2 ปีย้อนหลัง ==\n"
    try:
        markets = []
        c.execute("SELECT market, base_return_pct, bnh_return_pct, win_rate_pct, max_drawdown_pct FROM strategy_performance")
        for p in c.fetchall():
            perf_text += f"{p[0]}: ผลตอบแทนระบบ {p[1]:.2f}% (เทียบตลาด {p[2]:.2f}%), วินเรท {p[3]:.1f}%, DD สูงสุด {p[4]:.2f}%\n"
            markets.append(p[0])
        perf_text += "\n"
    except Exception:
        markets = ['BTC', 'US', 'UK', 'Thai', 'Gold']

    history_text = perf_text
    for market in markets:
        table_name = f"signals_history_{market}"
        try:
            c.execute(f'SELECT DISTINCT date FROM "{table_name}" ORDER BY date DESC LIMIT ?', (days,))
            dates = sorted(row[0] for row in c.fetchall())

            history_text += f"=== ข้อมูลตลาดย้อนหลัง {days} วัน: {market} ===\n"
            for d in dates:
                c.execute(f'SELECT market, trend_regime, signal_action, price, ml_up_prob, position FROM "{table_name}" WHERE date=?', (d,))
                for row in c.fetchall():
                    history_text += (
                        f"[{row[0]}] ราคา: {row[3]:.2f} | แนวโน้ม: {row[1]} | "
                        f"สัญญาณ: {row[2]} | ML ให้โอกาสขึ้น: {row[4]}% | "
                        f"พอร์ตถือของอยู่: {'Yes' if row[5]==1 else 'No'}\n"
                    )
            history_text += "\n"
        except sqlite3.OperationalError:
            pass

    conn.close()
    return history_text


def get_market_news_instruction() -> str:
    return (
        "ให้ใช้ความรู้ล่าสุดของคุณเกี่ยวกับข่าวเศรษฐกิจโลก, นโยบายธนาคารกลาง (FED/BOT/ECB), "
        "ภูมิรัฐศาสตร์, และเหตุการณ์สำคัญที่อาจส่งผลต่อตลาดการเงิน (หุ้น, ทอง, คริปโต) "
        "ในการประกอบการวิเคราะห์ร่วมกับข้อมูล Quantitative ด้านล่าง"
    )


def ask_llm(messages: list, model_override: str = None) -> Optional[str]:
    try:
        from openai import OpenAI
    except ImportError:
        return None

    if USE_OLLAMA:
        client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        model_name = model_override or OLLAMA_MODEL
    else:
        client = OpenAI(base_url=EXTERNAL_BASE_URL, api_key=EXTERNAL_API_KEY)
        model_name = model_override or EXTERNAL_MODEL

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM Error ({model_name}): {str(e)}")


def save_signals_to_sql_sync() -> dict:
    """Run trading_system.py --json and save results to SQLite."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "trading_system.py"), "--json"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    if result.returncode != 0:
        return {"success": False, "error": "trading_system.py failed", "stderr": result.stderr[:500]}

    try:
        signals_data = json.loads(result.stdout)
    except Exception as e:
        return {"success": False, "error": f"JSON parse error: {e}"}

    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    saved = []

    for m in signals_data:
        market = m['market']
        table_name = f"signals_history_{market}"

        c.execute(f'''
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT, market TEXT, price REAL,
                trend_regime TEXT, ml_up_prob REAL, ml_down_prob REAL,
                signal_action TEXT, position REAL,
                equity_curve REAL, bnh_curve REAL,
                UNIQUE(date)
            )
        ''')

        c.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE date=?', (today_date,))
        if c.fetchone()[0] == 0:
            trend_str = "1 (Uptrend)" if m.get('trend') == 'UPTREND' else "0 (Downtrend)"
            price = m.get('price', 0)
            if price == 'N/A':
                price = 0
                
            up_prob = m.get('ml_up_prob', 0.0)
            down_prob = m.get('ml_down_prob', 0.0)
            sig = m.get('signal', 99)

            equity_val = 0.0
            bnh_val = 0.0
            prev_pos = 0.0
            try:
                c.execute(f'SELECT equity_curve, bnh_curve, price, position FROM "{table_name}" WHERE equity_curve != 0 ORDER BY date DESC LIMIT 1')
                last_row = c.fetchone()
                if last_row and last_row[2] > 0:
                    prev_eq, prev_bnh, prev_price, prev_pos = last_row
                    price_ratio = price / prev_price if prev_price > 0 else 1.0
                    bnh_val = round(prev_bnh * price_ratio, 4)
                    if prev_pos > 0.0:
                        equity_val = round(prev_eq * price_ratio, 4)
                    elif prev_pos < 0.0:
                        equity_val = round(prev_eq * (2 - price_ratio), 4)
                    else:
                        equity_val = prev_eq
            except Exception:
                pass
                
            strat = m.get('strategy', 'active')
            if sig == 1:
                position = 1.0
            elif sig == 0:
                position = 0.0
            else:
                position = prev_pos if strat == 'smart_hold' else 0.0
                
            if position > 0 and prev_pos <= 0:
                sg_text = "BUY"
            elif position <= 0 and prev_pos > 0:
                sg_text = "SELL"
            elif position > 0 and prev_pos > 0:
                sg_text = "HOLD"
            else:
                sg_text = "WAIT"

            c.execute(f'''
                INSERT INTO "{table_name}"
                (date, market, price, trend_regime, ml_up_prob, ml_down_prob, signal_action, position, equity_curve, bnh_curve)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today_date, market, price, trend_str, up_prob, down_prob, sg_text, position, equity_val, bnh_val))
            saved.append(market)

    conn.commit()
    conn.close()
    return {"success": True, "saved_markets": saved, "date": today_date, "signals": signals_data}


# ─── Pydantic models ───

class ChatRequest(BaseModel):
    message: str
    history: list = []
    days: int = 5

class ChatResponse(BaseModel):
    response: str
    expert1: Optional[str] = None
    expert2: Optional[str] = None
    models_used: dict = {}


# ─── FastAPI App ───

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("✅ Database initialized")
    yield

app = FastAPI(
    title="AI Trading Assistant API",
    description="""
    LLM Hybrid Agent with Mixture-of-Experts — ระบบวิเคราะห์ตลาดด้วย AI
    
    API สำรองข้อมูลการเทรดและวิเคราะห์ทางเทคนิคสำหรับระบบ Trading Dashboard
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────────────────────────────────────
# Endpoints
# ───────────────────────────────────────

@app.get("/", tags=["System"], summary="API Root / Status")
async def root():
    return {
        "name": "AI Trading Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/chat": "Full MoE analysis (2 experts + master judge)",
            "POST /api/chat/simple": "Quick single-model response",
            "GET /api/signals": "Today's live trading signals",
            "GET /api/history": "SQL history (last N days)",
            "GET /api/performance": "Strategy performance stats (all markets)",
            "GET /api/markets": "List available markets",
            "POST /api/signals/save": "Save today's signals to DB",
        }
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["LLM Chat"], summary="Ask the MoE Trading Assistant")
async def chat_moe(req: ChatRequest):
    """
    Full Mixture-of-Experts analysis:
    Expert 1 (DeepSeek) + Expert 2 (Qwen) → Master Judge (Claude Haiku)
    """
    news_context = get_market_news_instruction()
    sql_history = get_sql_history(days=req.days)

    expert_prompt = f"""คำถามของผู้ใช้: {req.message}

[คำสั่งเกี่ยวกับข่าวสาร]:
{news_context}

[ข้อมูล Quantitative จาก Database (กลยุทธ์ 2 ปี ➕ สัญญาณ {req.days} วันล่าสุด)]:
{sql_history}

คำสั่ง:
ในฐานะผู้เชี่ยวชาญ Quant ให้หาตลาดที่ผู้ใช้ถามจากตาราง วิเคราะห์โดยอิงจากข่าวสารล่าสุดที่คุณรู้ + ข้อมูล Quantitative ข้างต้น แล้วสรุปสั้นๆ ว่าแนวโน้มเป็นยังไง, ML ให้โอกาสขึ้นกี่ %, และระบบมีสัญญาณ Buy/Sell/Hold อะไร"""

    # Run both experts concurrently
    loop = asyncio.get_event_loop()
    exp1_task = loop.run_in_executor(
        None, ask_llm, [{"role": "user", "content": expert_prompt}], EXPERT1_MODEL
    )
    exp2_task = loop.run_in_executor(
        None, ask_llm, [{"role": "user", "content": expert_prompt}], EXPERT2_MODEL
    )
    exp1_resp, exp2_resp = await asyncio.gather(exp1_task, exp2_task)

    # Master Judge synthesizes
    master_prompt = f"""คำถามของผู้ใช้: {req.message}

นี่คือความเห็นจากผู้เชี่ยวชาญ 2 ท่าน:
---
[Expert 1 ({EXPERT1_MODEL})]:
{exp1_resp}

---
[Expert 2 ({EXPERT2_MODEL})]:
{exp2_resp}
---

คำสั่งและกฎการตอบ (สำคัญมาก):
1. ในฐานะ **Master Quant** ให้สังเคราะห์ความเห็นจากผู้เชี่ยวชาญทั้งสอง
2. หน้าที่ของคุณคือ **สรุปแอคชั่นฟันธง (BUY/SELL/HOLD)** ออกมาให้สั้นที่สุด
3. **ห้ามพิมพ์คำว่า "บทสรุปแอคชั่นฟันธง"** หรืออารัมภบทใดๆ ทั้งสิ้น
4. **ห้ามระบุตัวเลขความมั่นใจ (Confidence %)** ไม่ต้องบอกว่ามั่นใจกี่เปอร์เซ็นต์ ให้ตอบสิ่งที่คุณเลือกและเหตุผลสั้นๆ ไม่เกิน 3 บรรทัด เช่น
"🚨 สัญญาณ: HOLD (รอ)
- ความเห็นผู้เชี่ยวชาญขัดแย้งกัน
- รอสัญญาณที่ชัดเจนกว่านี้"
"""

    messages = [
        {"role": "system", "content": "คุณเป็น Quantitative Analyst ระดับอาวุโส (AI Trading Assistant) ตอบเป็นภาษาไทย กระชับ ตรงประเด็น"},
        *req.history,
        {"role": "user", "content": master_prompt}
    ]

    master_resp = await loop.run_in_executor(None, ask_llm, messages, MASTER_MODEL)

    return ChatResponse(
        response=master_resp or "❌ ไม่สามารถเชื่อมต่อ LLM ได้",
        expert1=exp1_resp,
        expert2=exp2_resp,
        models_used={
            "expert1": EXPERT1_MODEL,
            "expert2": EXPERT2_MODEL,
            "master": MASTER_MODEL
        }
    )


@app.post("/api/chat/simple", tags=["LLM Chat"], summary="Quick Single-Model LLM Response")
async def chat_simple(req: ChatRequest):
    """Quick single-model response without MoE."""
    news_context = get_market_news_instruction()
    sql_history = get_sql_history(days=req.days)

    prompt = f"""คำถามของผู้ใช้: {req.message}

[คำสั่งเกี่ยวกับข่าวสาร]:
{news_context}

[ข้อมูล Quantitative จาก Database]:
{sql_history}

ตอบเป็นภาษาไทย กระชับ ฟันธงชัดเจน"""

    messages = [
        {"role": "system", "content": "คุณเป็น Quantitative Analyst ระดับอาวุโส ตอบเป็นภาษาไทย กระชับ ตรงประเด็น"},
        *req.history,
        {"role": "user", "content": prompt}
    ]

    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, ask_llm, messages, EXTERNAL_MODEL)

    return {
        "response": resp or "❌ ไม่สามารถเชื่อมต่อ LLM ได้",
        "model": EXTERNAL_MODEL
    }


@app.get("/api/signals", tags=["Live Trading"], summary="Get Today's Live Signals")
async def get_signals():
    """Get today's live trading signals from trading_system.py --json."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "trading_system.py"), "--json"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
    )

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail="trading_system.py failed")

    try:
        signals = json.loads(result.stdout)
        return {"date": datetime.datetime.now().strftime('%Y-%m-%d'), "signals": signals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")


@app.post("/api/signals/save", tags=["Live Trading"], summary="Force Save Today's Signals to Database")
async def save_signals():
    """Save today's signals to SQLite database."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, save_signals_to_sql_sync)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    return result


@app.get("/api/history", tags=["Historical Data"], summary="Get Signal History (Last N Days)")
async def get_history(
    days: int = Query(default=5, ge=1, le=365, description="Number of days to look back"),
    market: Optional[str] = Query(default=None, description="Filter by market (BTC, US, UK, Thai, Gold)")
):
    """Get signal history from SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    result = {}

    # Determine markets
    if market:
        markets = [market]
    else:
        try:
            c.execute("SELECT DISTINCT market FROM strategy_performance")
            markets = [row[0] for row in c.fetchall()]
        except Exception:
            markets = ['BTC', 'US', 'UK', 'Thai', 'Gold']

    for m in markets:
        table_name = f"signals_history_{m}"
        try:
            c.execute(f'SELECT * FROM "{table_name}" ORDER BY date DESC LIMIT ?', (days,))
            rows = [dict(row) for row in c.fetchall()]
            result[m] = rows
        except sqlite3.OperationalError:
            result[m] = []

    conn.close()
    return {"days": days, "markets": result}


@app.get("/api/performance", tags=["Historical Data"], summary="Get Backtested Strategy Performance")
async def get_performance():
    """Get strategy performance stats for all markets."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        c.execute("SELECT * FROM strategy_performance ORDER BY market")
        rows = [dict(row) for row in c.fetchall()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

    return {"markets": rows}


@app.get("/api/markets", tags=["System"], summary="List Supported Markets")
async def get_markets():
    """List available markets with their current status."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    markets = []
    try:
        c.execute("SELECT market, base_return_pct, bnh_return_pct, win_rate_pct, max_drawdown_pct, updated_at FROM strategy_performance")
        for row in c.fetchall():
            markets.append({
                "market": row[0],
                "strategy_return": row[1],
                "bnh_return": row[2],
                "vs_bnh": round(row[1] - row[2], 2) if row[1] and row[2] else None,
                "beats_bnh": row[1] > row[2] if row[1] and row[2] else False,
                "win_rate": row[3],
                "max_drawdown": row[4],
                "last_updated": row[5]
            })
    except Exception:
        pass
    finally:
        conn.close()

    return {"total": len(markets), "markets": markets}


@app.get("/api/data")
async def get_dashboard_data(market: str = Query(default="BTC", description="Market name (e.g. BTC, US, Thai)")):
    """Get full combined stats and history for a specific market (for frontend dashboards like Flutter)."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        # Fetch Overview Stats
        c.execute("SELECT * FROM strategy_performance WHERE market = ?", (market,))
        stats_row = c.fetchone()
        
        # Fetch Daily Signals
        table_name = f"signals_history_{market}"
        c.execute(f'SELECT * FROM "{table_name}" ORDER BY date ASC')
        history_rows = c.fetchall()
        
        return {
            "market": market,
            "stats": dict(stats_row) if stats_row else None,
            "history": [dict(r) for r in history_rows]
        }
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=404, detail=f"Market data not found: {e}")
    finally:
        conn.close()


@app.get("/api/signals/markers", tags=["Live Trading"], summary="Get Buy/Sell Signal Markers for Dashboard")
async def get_signal_markers(
    market: str = Query(default="BTC", description="Market name"),
    limit: int = Query(default=20, ge=1, le=100, description="Max number of signal markers to return")
):
    """
    Return only the actual position-change signals (BUY/SELL) for overlay on charts.
    Also returns the latest/current signal status.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        table_name = f"signals_history_{market}"
        c.execute(f'SELECT date, price, position, signal_action, trend_regime, ml_up_prob, ml_down_prob FROM "{table_name}" ORDER BY date ASC')
        rows = c.fetchall()

        markers = []
        prev_pos = 0.0
        latest_signal = None

        for row in rows:
            r = dict(row)
            curr_pos = float(r.get('position', 0))
            action = r.get('signal_action', 'WAIT')

            latest_signal = {
                "date": r['date'],
                "price": r['price'],
                "action": action,
                "position": curr_pos,
                "trend": r.get('trend_regime', ''),
                "ml_up_prob": r.get('ml_up_prob', 0),
                "ml_down_prob": r.get('ml_down_prob', 0),
            }

            # Only record actual position changes (BUY or SELL)
            if curr_pos != prev_pos:
                marker_type = "BUY" if curr_pos > 0 else "SELL"
                markers.append({
                    "date": r['date'],
                    "price": r['price'],
                    "type": marker_type,
                    "action": action,
                    "ml_up_prob": r.get('ml_up_prob', 0),
                })
            prev_pos = curr_pos

        # Return most recent markers (limit)
        recent_markers = markers[-limit:] if len(markers) > limit else markers

        return {
            "market": market,
            "current_signal": latest_signal,
            "total_signals": len(markers),
            "markers": recent_markers,
        }

    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=404, detail=f"Market data not found: {e}")
    finally:
        conn.close()


# ─── Run with: python workflows/api_server.py ───
if __name__ == "__main__":
    import uvicorn
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print("🚀 AI Trading Assistant API starting...")
    print("📖 Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
