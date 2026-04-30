import sqlite3
import json
import os
import sys
import datetime

# Ensure workflows/ directory is on sys.path so trading_system can be imported
_workflows_dir = os.path.dirname(os.path.abspath(__file__))
if _workflows_dir not in sys.path:
    sys.path.insert(0, _workflows_dir)

# ---------------------------------------------------------
# ตั้งค่า LLM (Ollama / API) เสมือน llm_agent.py
# ---------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: ไม่พบแพ็กเกจ 'python-dotenv' ระบบอาจโหลด API Key จาก .env ไม่ได้ (ข้ามไปใช้ Environment Variables แทน)")

USE_OLLAMA = False
OLLAMA_MODEL = "llama3.2:1b"
EXTERNAL_MODEL = "minimax/minimax-01"  # Best available model for complex reasoning and consensus
EXTERNAL_API_KEY = os.getenv("OPENCODE_API_KEY", "YOUR_API_KEY")
EXTERNAL_BASE_URL = os.getenv("OPENCODE_BASE_URL", "https://api.minimax.chat/v1")

DB_FILE = os.path.join(os.path.dirname(__file__), "trading_database.sqlite")

# =========================================================
# 1. ฐานข้อมูล SQL (เก็บสถิติสัญญาณรายวัน)
# =========================================================
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

def catch_up_missing_days(trading_system_module):
    """เติมข้อมูลย้อนหลังจากวันที่ล่าสุดใน DB ให้ถึงวันปัจจุบัน"""
    markets = getattr(trading_system_module, 'MARKETS', ['US', 'UK', 'Thai', 'Gold', 'BTC'])
    today = datetime.datetime.now().date()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    missing_dates = set()

    for market in markets:
        table_name = f"signals_history_{market}"
        try:
            cursor.execute(
                f'SELECT date FROM "{table_name}" '
                'WHERE (equity_curve IS NULL OR equity_curve=0) '
                'AND (bnh_curve IS NULL OR bnh_curve=0)'
            )
            bad_rows = [r[0] for r in cursor.fetchall() if r[0]]
            if bad_rows:
                cursor.executemany(f'DELETE FROM "{table_name}" WHERE date=?', [(d,) for d in bad_rows])
                conn.commit()
                for bad in bad_rows:
                    bad_date = datetime.datetime.strptime(bad, '%Y-%m-%d').date()
                    missing_dates.add(bad_date)
                print(f"⚠️ ลบข้อมูลเสีย {table_name} จำนวน {len(bad_rows)} แถว (eq/bnh = 0)")

            cursor.execute(f'SELECT date FROM "{table_name}" ORDER BY date ASC')
            existing_rows = [r[0] for r in cursor.fetchall() if r[0]]
        except sqlite3.OperationalError:
            continue

        if not existing_rows:
            continue

        existing_set = {
            datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in existing_rows
        }
        first_date = min(existing_set)
        cur_day = first_date + datetime.timedelta(days=1)
        while cur_day < today:
            if cur_day not in existing_set:
                missing_dates.add(cur_day)
            cur_day += datetime.timedelta(days=1)

    conn.close()

    if not missing_dates:
        return

    for day in sorted(missing_dates):
        date_str = day.strftime('%Y-%m-%d')
        # print(f"⏪ กำลังเติมข้อมูลย้อนหลัง {date_str} ...")
        try:
            results = trading_system_module.get_current_signals(quiet=True, as_of_date=date_str)
        except Exception as e:
            print(f"❌ ดึงสัญญาณสำหรับ {date_str} ไม่สำเร็จ: {e}")
            break

        valid_results = []
        for r in results:
            price = r.get('price')
            try:
                price_val = float(price)
            except (TypeError, ValueError):
                continue
            if price_val <= 0:
                continue
            valid_results.append(r)
        if not valid_results:
            continue

        saved = trading_system_module.save_signals_to_db(valid_results, quiet=True)
        # if saved:
        #     print(f"   ↳ เติมข้อมูลวันที่ {date_str} แล้ว ({saved} แถว)")

def save_signals_to_sql():
    """
    ดึงสัญญาณล่าสุดจาก trading_system โดยตรงและใช้ฟังก์ชัน save_signals_to_db
    เพื่อให้ข้อมูล (ml_up_prob, equity_curve ฯลฯ) ถูกต้องเหมือนที่ระบบหลักใช้
    """
    try:
        import trading_system
    except ImportError as e:
        print("❌ นำเข้า trading_system.py ไม่ได้:", e)
        return False

    catch_up_missing_days(trading_system)

    try:
        results = trading_system.get_current_signals(quiet=True)
    except Exception as e:
        print("❌ ดึงสัญญาณล่าสุดไม่สำเร็จ:", e)
        return False

    valid_results = []
    for m in results:
        price = m.get('price')
        try:
            price_val = float(price)
        except (TypeError, ValueError):
            continue
        if price_val <= 0:
            continue
        valid_results.append(m)
    if not valid_results:
        print("⚠️ ไม่มีข้อมูลราคาที่ใช้งานได้จาก trading_system (อาจเป็นวันหยุดหรือตลาดปิด)")
        return False

    try:
        trading_system.save_signals_to_db(valid_results, quiet=True)
        # print("✅ บันทึกสัญญาณล่าสุดลง SQLite แล้ว")
        return True
    except Exception as e:
        print("❌ บันทึกลง SQLite ไม่สำเร็จ:", e)
        return False

def get_sql_history(days=5):
    """อ่านข้อมูลย้อนหลัง N วัน จาก SQL แบบละเอียด"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # โหลดภาพรวม Strategy Performance เพื่อให้ LLM รู้จักพื้นฐาน
    perf_text = "== ภาพรวมประสิทธิภาพระบบเทรด 2 ปีย้อนหลัง ==\n"
    try:
        markets = []
        c.execute("SELECT market, base_return_pct, bnh_return_pct, win_rate_pct, max_drawdown_pct FROM strategy_performance")
        for p in c.fetchall():
            perf_text += f"{p[0]}: ผลตอบแทนระบบ {p[1]:.2f}% (เทียบตลาด {p[2]:.2f}%), วินเรท {p[3]:.1f}%, DD สูงสุด {p[4]:.2f}%\n"
            markets.append(p[0])
        perf_text += "\n"
    except Exception:
        markets = ['BTC', 'US', 'UK', 'Thai', 'Gold'] # Fallback
        pass

    history_text = perf_text
    for market in markets:
        table_name = f"signals_history_{market}"
        try:
            c.execute(f'SELECT DISTINCT date FROM "{table_name}" ORDER BY date DESC LIMIT ?', (days,))
            dates = [row[0] for row in c.fetchall()]
            dates.sort()
            
            history_text += f"=== ข้อมูลตลาดย้อนหลัง {days} วัน: {market} ===\n"
            for d in dates:
                c.execute(f'SELECT market, trend_regime, signal_action, price, ml_up_prob, position FROM "{table_name}" WHERE date=?', (d,))
                for row in c.fetchall():
                    history_text += f"[{row[0]}] ราคา: {row[3]:.2f} | แนวโน้ม: {row[1]} | สัญญาณ: {row[2]} | ML ให้โอกาสขึ้น: {row[4]}% | พอร์ตถือของอยู่: {'Yes' if row[5]==1 else 'No'}\n"
            history_text += "\n"
        except sqlite3.OperationalError:
            pass
        
    conn.close()
    return history_text

# =========================================================
# 2. ข่าวสารระดับมหภาค (LLM Native Knowledge)
# =========================================================
def get_market_news_instruction():
    """สั่งให้ LLM ใช้ความรู้ตัวเองเกี่ยวกับการวิเคราะห์โดยไม่แต่งเรื่อง"""
    return """ในการวิเคราะห์ ให้พิจารณาเฉพาะข้อมูล Quantitative ที่ได้รับเป็นหลัก 
ห้ามแต่งเรื่อง เดาสุ่ม หรือยกเหตุผลเชิงเศรษฐกิจมหภาคทั่วไป (เช่น การประชุม FED, ตัวเลข GDP, หรือสงคราม) มาอ้างอิงเด็ดขาดเว้นแต่จะระบุชัดเจนในคำถาม
ให้วิเคราะห์ความเสี่ยงจากตัวเลขสถิติ, ความผันผวนของราคา, โอกาสทางสถิติ (ML prob), และ Max Drawdown เท่านั้น"""

# =========================================================
# 3. LLM Hybrid Prompt (SQL + RAG)
# =========================================================
def ask_llm(messages, model_override=None):
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ ไม่พบไลบรารี openai")
        return None

    if USE_OLLAMA:
        client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        model_name = model_override if model_override else OLLAMA_MODEL
    else:
        client = OpenAI(base_url=EXTERNAL_BASE_URL, api_key=EXTERNAL_API_KEY)
        model_name = model_override if model_override else EXTERNAL_MODEL

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาดในการเชื่อมต่อ LLM ({model_name}): {e}")
        return None

# =========================================================
# 4. Casual message detection
# =========================================================
CASUAL_PATTERNS = [
    '\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35', '\u0e2b\u0e27\u0e31\u0e14\u0e14\u0e35', '\u0e14\u0e35\u0e04\u0e23\u0e31\u0e1a', '\u0e14\u0e35\u0e04\u0e48\u0e30', '\u0e14\u0e35\u0e08\u0e49\u0e32', '\u0e14\u0e35\u0e08\u0e49\u0e30', '\u0e14\u0e35\u0e19\u0e30',
    '\u0e40\u0e1b\u0e47\u0e19\u0e44\u0e07', '\u0e40\u0e1b\u0e47\u0e19\u0e22\u0e31\u0e07\u0e44\u0e07', '\u0e17\u0e33\u0e2d\u0e30\u0e44\u0e23\u0e2d\u0e22\u0e39\u0e48', '\u0e27\u0e48\u0e32\u0e44\u0e07', '\u0e44\u0e07',
    '\u0e02\u0e2d\u0e1a\u0e04\u0e38\u0e13', '\u0e02\u0e2d\u0e1a\u0e43\u0e08',
    '\u0e25\u0e32\u0e01\u0e48\u0e2d\u0e19', '\u0e1a\u0e32\u0e22', '\u0e44\u0e1b\u0e41\u0e25\u0e49\u0e27', '\u0e44\u0e1b\u0e01\u0e48\u0e2d\u0e19', '\u0e44\u0e1b\u0e25\u0e30',
    '\u0e04\u0e38\u0e13\u0e40\u0e1b\u0e47\u0e19\u0e43\u0e04\u0e23', '\u0e17\u0e33\u0e2d\u0e30\u0e44\u0e23\u0e44\u0e14\u0e49\u0e1a\u0e49\u0e32\u0e07', '\u0e0a\u0e48\u0e27\u0e22\u0e2d\u0e30\u0e44\u0e23\u0e44\u0e14\u0e49\u0e1a\u0e49\u0e32\u0e07', '\u0e04\u0e38\u0e13\u0e04\u0e37\u0e2d\u0e43\u0e04\u0e23',
    '\u0e2d\u0e23\u0e38\u0e13\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e34\u0e4c', '\u0e23\u0e32\u0e15\u0e23\u0e35\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e34\u0e4c', '\u0e19\u0e2d\u0e19\u0e2b\u0e25\u0e31\u0e1a\u0e1d\u0e31\u0e19\u0e14\u0e35',
    'hello', 'hi', 'hey', 'yo', 'sup',
    'good morning', 'good evening', 'good night',
    'how are you', "what's up", 'whats up',
    'thank you', 'thanks', 'bye', 'goodbye',
    'who are you', 'what can you do',
]

# Keywords that indicate a trading/market question (override casual detection)
TRADING_KEYWORDS = [
    # Thai market/trading terms
    '\u0e23\u0e32\u0e04\u0e32', '\u0e15\u0e25\u0e32\u0e14', '\u0e2b\u0e38\u0e49\u0e19', '\u0e17\u0e2d\u0e07', '\u0e17\u0e2d\u0e07\u0e04\u0e33',
    '\u0e40\u0e17\u0e23\u0e14', '\u0e25\u0e07\u0e17\u0e38\u0e19', '\u0e1e\u0e2d\u0e23\u0e4c\u0e15', '\u0e2a\u0e31\u0e0d\u0e0d\u0e32\u0e13',
    '\u0e0b\u0e37\u0e49\u0e2d', '\u0e02\u0e32\u0e22', '\u0e16\u0e37\u0e2d', '\u0e02\u0e36\u0e49\u0e19', '\u0e25\u0e07',
    '\u0e41\u0e19\u0e27\u0e42\u0e19\u0e49\u0e21', '\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c', '\u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33',
    '\u0e2d\u0e31\u0e15\u0e23\u0e32\u0e41\u0e25\u0e01\u0e40\u0e1b\u0e25\u0e35\u0e48\u0e22\u0e19', '\u0e1c\u0e25\u0e15\u0e2d\u0e1a\u0e41\u0e17\u0e19',
    '\u0e04\u0e27\u0e32\u0e21\u0e40\u0e2a\u0e35\u0e48\u0e22\u0e07', '\u0e01\u0e33\u0e44\u0e23',
    # English market terms
    'btc', 'bitcoin', 'gold', 'stock', 'market',
    'buy', 'sell', 'hold', 'trade', 'price',
    'signal', 'trend', 'crypto', 'forex',
    'us', 'uk', 'thai', 'sp500', 's&p',
    'bull', 'bear', 'portfolio',
]

def is_casual_message(text):
    cleaned = text.strip().lower()
    # If message contains trading keywords, it's NOT casual
    for kw in TRADING_KEYWORDS:
        if kw in cleaned:
            return False
    # Short messages matching casual patterns are casual
    if len(cleaned) <= 40:
        for pattern in CASUAL_PATTERNS:
            if pattern in cleaned:
                return True
    return False

def handle_casual_response(user_input):
    casual_system = {"role": "system", "content": "\u0e04\u0e38\u0e13\u0e40\u0e1b\u0e47\u0e19 AI Trading Assistant \u0e17\u0e35\u0e48\u0e40\u0e1b\u0e47\u0e19\u0e01\u0e31\u0e19\u0e40\u0e2d\u0e07\n\u0e40\u0e21\u0e37\u0e48\u0e2d\u0e1c\u0e39\u0e49\u0e43\u0e0a\u0e49\u0e17\u0e31\u0e01\u0e17\u0e32\u0e22\u0e2b\u0e23\u0e37\u0e2d\u0e04\u0e38\u0e22\u0e40\u0e25\u0e48\u0e19 \u0e43\u0e2b\u0e49\u0e15\u0e2d\u0e1a\u0e2a\u0e31\u0e49\u0e19\u0e46 \u0e40\u0e1b\u0e47\u0e19\u0e01\u0e31\u0e19\u0e40\u0e2d\u0e07 \u0e2d\u0e1a\u0e2d\u0e38\u0e48\u0e19 \u0e20\u0e32\u0e29\u0e32\u0e44\u0e17\u0e22\n\u0e41\u0e19\u0e30\u0e19\u0e33\u0e15\u0e31\u0e27\u0e2a\u0e31\u0e49\u0e19\u0e46 \u0e27\u0e48\u0e32\u0e0a\u0e48\u0e27\u0e22\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c\u0e15\u0e25\u0e32\u0e14\u0e41\u0e25\u0e30\u0e43\u0e2b\u0e49\u0e2a\u0e31\u0e0d\u0e0d\u0e32\u0e13\u0e40\u0e17\u0e23\u0e14\u0e44\u0e14\u0e49\n\u0e2b\u0e49\u0e32\u0e21\u0e43\u0e2b\u0e49\u0e04\u0e33\u0e41\u0e19\u0e30\u0e19\u0e33\u0e40\u0e17\u0e23\u0e14\u0e2b\u0e23\u0e37\u0e2d\u0e27\u0e34\u0e40\u0e04\u0e23\u0e32\u0e30\u0e2b\u0e4c\u0e15\u0e25\u0e32\u0e14\u0e43\u0e19\u0e01\u0e32\u0e23\u0e17\u0e31\u0e01\u0e17\u0e32\u0e22\n\u0e15\u0e2d\u0e1a 1-2 \u0e1b\u0e23\u0e30\u0e42\u0e22\u0e04\u0e1e\u0e2d"}
    messages = [casual_system, {"role": "user", "content": user_input}]
    return ask_llm(messages, model_override="minimax/minimax-01")

if __name__ == "__main__":
    init_db()

    
    print("\n--- 1. ระบบ SQL Database ---")
    print("กำลังดึงและบันทึกสัญญาณวันนี้...")
    save_signals_to_sql()
    
    print("\n=======================================================")
    print("🤖 ยินดีต้อนรับสู่ AI Trading Assistant (Hybrid SQL+RAG)")
    print("พิมพ์คำถามที่คุณอยากวิเคราะห์ (พิมพ์ 'exit' เพื่อออก)")
    print("=======================================================\n")
    
    chat_history = [
        {"role": "system", "content": """คุณเป็น Quantitative Analyst ระดับอาวุโส (AI Trading Assistant) หน้าที่หลักคล้ายที่ปรึกษาการลงทุน 
กฎเหล็กในการตอบ:
1. ห้ามแค่ท่อง 'ภาพรวมผลงาน 2 ปี' หรือลอกค่า Stat กลับมาเฉยๆ! 
2. ให้วิเคราะห์ลึกไปถึง 'สถิติรายวัน 5 วันล่าสุด' ของตลาดที่ถูกถาม (Regime เป็นเทรนด์ไหน? ML มั่นใจกี่เปอร์เซ็นต์? สัญญาณให้ HOLD หรือ BUY?)
3. ฟันธงข้อสรุปให้ชัดเจนว่า "ควรแอคชั่นอย่างไร" โดยอิงตัวเลขความน่าจะเป็นล่าสุด ไม่ใช่ความรู้สึก
4. ตอบเป็นภาษาไทย กระชับ ตรงประเด็นแบบมืออาชีพ
"""}
    ]
    
    while True:
        try:
            user_input = input("👤 คุณ: ")
            user_input = user_input.encode('utf-8', 'replace').decode('utf-8')
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("ลาก่อน! ขอให้เทรดได้กำไรเยอะๆ")
            break
            
        if not user_input.strip():
            continue

        # ----- Casual message shortcut (greetings/chitchat) -----
        if is_casual_message(user_input):
            response_text = handle_casual_response(user_input)
            if response_text:
                print(f"\n🤖 AI: {response_text}\n")
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response_text})
            continue

        print("🔍 กำลังค้นหาข้อมูลที่เกี่ยวข้องจาก Database...")
        news_context = get_market_news_instruction()
        sql_history = get_sql_history(days=5)
        
        # -------------------------------------------------------------
        # กะเกณฑ์โมเดลผู้เชี่ยวชาญ (MoE - Mixture of Experts) - โหมด Turbo เร็วปรื๊ด
        # -------------------------------------------------------------
        expert1_model = "deepseek/deepseek-r1-distill-llama-70b"
        expert2_model = "qwen/qwen-2.5-72b-instruct"
        master_model = "minimax/minimax-01"
        
        expert_prompt = f"""คำถามของผู้ใช้: {user_input}

[คำสั่งเกี่ยวกับข่าวสาร]:
{news_context}

[ข้อมูล Quantitative จาก Database (กลยุทธ์ 2 ปี ➕ สัญญาณ 5 วันล่าสุด)]:
{sql_history}

คำสั่ง: 
ในฐานะผู้เชี่ยวชาญ Quant ให้หาตลาดที่ผู้ใช้ถามจากตาราง วิเคราะห์โดยอิงจากข่าวสารล่าสุดที่คุณรู้ + ข้อมูล Quantitative ข้างต้น แล้วสรุปสั้นๆ ว่าแนวโน้มเป็นยังไง, ML ให้โอกาสขึ้นกี่ %, และระบบมีสัญญาณ Buy/Sell/Hold อะไร"""

        print(f"🧠 Expert 1 ({expert1_model}) กำลังวิเคราะห์อย่างอิสระ...")
        exp1_resp = ask_llm([{"role": "user", "content": expert_prompt}], model_override=expert1_model)
        
        print(f"🧠 Expert 2 ({expert2_model}) กำลังวิเคราะห์อย่างอิสระ...")
        exp2_resp = ask_llm([{"role": "user", "content": expert_prompt}], model_override=expert2_model)

        master_prompt = f"""คำถามของผู้ใช้: {user_input}

นี่คือความเห็นจากผู้เชี่ยวชาญ 2 ท่าน:
---
[Expert 1 ({expert1_model})]:
{exp1_resp}

---
[Expert 2 ({expert2_model})]:
{exp2_resp}
---

คำสั่งและกฎการตอบ (สำคัญมาก):
1. ในฐานะ **Master Quant** ให้สังเคราะห์ความเห็นจากผู้เชี่ยวชาญทั้งสอง
2. ฟันธง **BUY / SELL / HOLD** โดยอาศัยข้อมูลล่าสุด ไม่ใช่ความรู้สึก
3. ใช้ภาษาไทยแบบมนุษย์ เป็นกันเอง เข้าใจง่าย และเลี่ยงศัพท์เทคนิคที่ไม่จำเป็น
4. ห้ามระบุเปอร์เซ็นต์ความมั่นใจ
5. ห้ามอ้างอิงถึง FED, ดอกเบี้ย, GDP หรือข่าวเศรษฐกิจเว้นแต่จะมีระบุในข้อมูล (ห้ามแต่งเรื่อง)
6. รูปแบบคำตอบ:
   - บรรทัดแรกขึ้นต้นด้วย `คำแนะนำ: <BUY/SELL/HOLD> – ...` พร้อมเหตุผลย่อ
   - ตามด้วย 2 bullet ย่อยที่ขยายเหตุผลหลักและสิ่งที่ควรจับตา (ถ้าพูดถึงตลาดอื่น เช่น Gold หรือ BTC ให้รวมไว้ใน bullet)
   - แต่ละ bullet เป็นประโยคสั้นที่เชื่อมโยงกับข้อมูลเชิงตัวเลขหรือสัญญาณในฐานข้อมูล"""
        
        # Don't keep piling up system prompts, just append the user request
        chat_history.append({"role": "user", "content": master_prompt})
        
        print(f"👑 Master Judge ({master_model}) กำลังฟันธงผลลัพธ์...\n")
        response_text = ask_llm(chat_history, model_override=master_model)
        
        if response_text:
            print(f"🤖 AI: {response_text}\n")
            # บันทึกคำตอบกลับเข้าไปใน History (แต่เพื่อให้ Context ไม่ล้น เราจะเก็บคำถาม+คำตอบแบบธรรมดา)
            # แก้ role user กลับเป็นเนื้อหาจริงๆ (ลบ context ทิ้งใน history เพื่อไม่ให้เปลือง Token)
            chat_history[-1]["content"] = user_input 
            chat_history.append({"role": "assistant", "content": response_text})
        else:
            chat_history.pop() # ลบคำถามออกถ้าเชื่อมต่อไม่ได้
