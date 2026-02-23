import sqlite3
import json
import os
import datetime
import subprocess

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
EXTERNAL_MODEL = "anthropic/claude-3.5-sonnet"  # Best available model for complex reasoning and consensus
EXTERNAL_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_API_KEY")
EXTERNAL_BASE_URL = "https://openrouter.ai/api/v1"

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
    c.execute('''
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_signals_to_sql():
    """ดึงข้อมูล --json จาก trading_system.py แล้ว Save ลง SQLite แบบ Schema ใหม่"""
    result = subprocess.run(
        ["python3", "workflows/trading_system.py", "--json"], 
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("❌ เกิดข้อผิดพลาดในการดึงสัญญาณ")
        return False
        
    try:
        signals_data = json.loads(result.stdout)
    except Exception as e:
        print("❌ แปลง JSON ไม่สำเร็จ:", e)
        return False

    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    for m in signals_data:
        market = m['market']
        table_name = f"signals_history_{market}"
        
        c.execute(f'''
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                market TEXT,
                price REAL,
                trend_regime TEXT,
                ml_up_prob REAL,
                ml_down_prob REAL,
                signal_action TEXT,
                position REAL,
                equity_curve REAL,
                bnh_curve REAL,
                UNIQUE(date)
            )
        ''')
        
        # Check if we already have today's insert
        c.execute(f'SELECT COUNT(*) FROM "{table_name}" WHERE date=?', (today_date,))
        if c.fetchone()[0] == 0:
            trend_str = "1 (Uptrend)" if m['trend'] == 'UPTREND' else "0 (Downtrend)"
            price = m.get('price', 0)
            if price == 'N/A': price = 0
            
            c.execute(f'''
                INSERT INTO "{table_name}" 
                (date, market, price, trend_regime, ml_up_prob, ml_down_prob, signal_action, position, equity_curve, bnh_curve)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (today_date, market, price, trend_str, 0, 0, m['signal_text'], 0, 0, 0))
            print(f"✅ บันทึกสัญญาณของวันที่ {today_date} ลง SQL ({table_name}) เรียบร้อย")
        else:
            print(f"ℹ️ สัญญาณของวันที่ ({today_date}) ถูกบันทึกไว้ใน SQL ({table_name}) แล้ว")
            
    conn.commit()
    conn.close()
    return True

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
# 2. RAG System (เก็บและค้นหาข่าว) ด้วย Scikit-Learn
# =========================================================
def add_news_to_rag(date, content):
    """บันทึกข่าวลง Database สำหรับทำ RAG"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO news_articles (date, content) VALUES (?, ?)", (date, content))
    conn.commit()
    conn.close()
    print(f"📰 เพิ่มข่าวของวันที่ {date} ลง RAG Database เรียบร้อย")

def retrieve_relevant_news(query, top_k=2):
    """รันระบบ Vector Search (TF-IDF Cosine Similarity) หาข่าวที่เกี่ยวกับสิ่งที่ถาม"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("❌ ไม่พบ scikit-learn (ไม่สามารถรันระบบ RAG ได้)")
        return "ไม่มีข้อมูลข่าวสาร (ไม่พบไลบรารี sklearn)"

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT date, content FROM news_articles ORDER BY id DESC LIMIT 50")  # ดึงข่าว 50 ล่าสุด
    articles = c.fetchall()
    conn.close()
    
    if not articles:
        return "ไม่มีข้อมูลข่าวสารในระบบ"
        
    docs = [row[1] for row in articles]
    docs_with_dates = [f"[{row[0]}] {row[1]}" for row in articles]
    
    # Vectorize (แปลงข้อความเป็นตัวเลขเพื่อหาความสอดคล้อง)
    vectorizer = TfidfVectorizer()
    all_text = [query] + docs
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    # เทียบความคล้ายคลังเวกเตอร์
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    related_docs_indices = cosine_similarities.argsort()[-top_k:][::-1] # เอา top_k ที่มีคะแนนสูงสุด
    
    rag_context = ""
    for idx in related_docs_indices:
        if cosine_similarities[idx] > 0.05: # ถัามีความคล้ายเกิน 5% ค่อยเอามา
            rag_context += f"- {docs_with_dates[idx]}\n"
            
    if not rag_context:
        return "ไม่มีข่าวที่เกี่ยวข้องกับเหตุการณ์ล่าสุดโดยตรง"
    return rag_context

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

if __name__ == "__main__":
    init_db()
    
    print("\n--- 1. ระบบ SQL Database ---")
    print("📡 กำลังดึงและบันทึกสัญญาณวันนี้...")
    save_signals_to_sql()
    
    # -------------------------------------------------------------
    # ระบบจำลอง (Mock): ลองป้อนข่าวเข้า RAG
    # -------------------------------------------------------------
    print("\n--- 2. ระบบ RAG VectorDB ---")
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    
    conn = sqlite3.connect(DB_FILE)
    count_news = conn.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]
    conn.close()
    
    if count_news == 0:
        add_news_to_rag(yesterday, "กนง. ประเทศไทย (BOT) มีมติคงอัตราดอกเบี้ยนโยบายไว้ที่ 2.50% ทำให้ตลาดหุ้นไทยแกว่งตัวแคบ ๆ ขาดปัจจัยบวกใหม่เข้ามาหนุน")
        add_news_to_rag(today, "เงินเฟ้อสหรัฐพุ่งสูงขึ้นกว่าที่คาดการณ์ ทำให้นักลงทุนกังวลว่า FED อาจจะคงดอกเบี้ยสูงไปอีกนาน ส่งผลลบต่อ Bitcoin (BTC) และสินทรัพย์เสี่ยง (US Stocks)")
        add_news_to_rag(today, "ทองคำ (Gold) ได้รับอานิสงส์ความตึงเครียดทางภูมิรัฐศาสตร์เป็นที่พักพิงที่ปลอดภัยในช่วงนี้")
    
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
            print("ลาก่อน! ขอให้เทรดได้กำไรเยอะๆ 🚀")
            break
            
        if not user_input.strip():
            continue
            
        print("🔍 กำลังค้นหาข้อมูลที่เกี่ยวข้องจาก Database...")
        news_context = retrieve_relevant_news(user_input)
        sql_history = get_sql_history(days=5)
        
        # -------------------------------------------------------------
        # กะเกณฑ์โมเดลผู้เชี่ยวชาญ (MoE - Mixture of Experts) - โหมด Turbo เร็วปรื๊ด
        # -------------------------------------------------------------
        expert1_model = "deepseek/deepseek-r1-distill-llama-70b" # รุ่นบีบอัดของ R1 คิดไว ตอบเร็ว
        expert2_model = "qwen/qwen-2.5-72b-instruct" # รุ่นล่าสุดของ Qwen ที่เบาและไวกว่า Max มาก
        master_model = "anthropic/claude-3-5-haiku" # รุ่นน้องเล็กสุดของ Claude 3.5 ที่เน้นความไวแสง
        
        expert_prompt = f"""คำถามของผู้ใช้: {user_input}

[ข้อมูล RAG ข่าวสารระดับมหภาค]:
{news_context}

[ข้อมูล Quantitative จาก Database (กลยุทธ์ 2 ปี ➕ สัญญาณ 5 วันล่าสุด)]:
{sql_history}

คำสั่ง: 
ในฐานะผู้เชี่ยวชาญ Quant ให้หาตลาดที่ผู้ใช้ถามจากตาราง แล้ววิเคราะห์สั้นๆ ว่าแนวโน้มเป็นยังไง, ML ให้โอกาสขึ้นกี่ %, และระบบมีสัญญาณ Buy/Sell/Hold อะไร"""

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
2. หน้าที่ของคุณคือ **สรุปแอคชั่นฟันธง (BUY/SELL/HOLD)** ออกมาให้สั้นที่สุด
3. **ห้ามพิมพ์คำว่า "บทสรุปแอคชั่นฟันธง"** หรืออารัมภบทใดๆ ทั้งสิ้น
4. **ห้ามระบุตัวเลขความมั่นใจ (Confidence %)** ไม่ต้องบอกว่ามั่นใจกี่เปอร์เซ็นต์ ให้ตอบสิ่งที่คุณเลือกและเหตุผลสั้นๆ ไม่เกิน 3 บรรทัด เช่น
"🚨 สัญญาณ: HOLD (รอ)
- ความเห็นผู้เชี่ยวชาญขัดแย้งกัน
- รอสัญญาณที่ชัดเจนกว่านี้"
"""
        
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
