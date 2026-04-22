# AI Trading Assistant (Hybrid SQL+RAG)

คู่มือการใช้งานและรันระบบ AI Trading Assistant สำหรับวิเคราะห์ข้อมูลตลาดหลักทรัพย์แบบ Realtime

## 🚀 วิธีการรันระบบ

การรันระบบมี 2 รูปแบบหลักๆ ให้เลือกใช้ตามสภาพแวดล้อม (Environment) ของเครื่องคุณ:

### วิธีที่ 1: รันแบบปกติ (แนะนำ)
หากคุณไม่ได้เปิดใช้งาน Conda หรือแอพที่รบกวน Python Path คุณสามารถใช้คำสั่งปกติได้เลย:

```bash
cd /Users/oattao/project/p-e
./run.sh
```

### วิธีที่ 2: รันเมื่อมี Conda (แก้ปัญหา Numpy Error)
หากใช้เทอร์มินัลที่มี Conda รันอยู่ (สังเกตจากคำว่า `base` หรือชื่อ environment อื่นๆ ใน prompt) Conda อาจไปแทรกแซง Path ของไลบรารีจนทำให้เกิด Error: `Error importing numpy: you should not try to import numpy from its source directory`

ให้รันด้วยคำสั่งนี้แทนเพื่อบังคับเคลียร์ Path ชั่วคราว:
```bash
cd /Users/oattao/project/p-e
env -u PYTHONHOME -u PYTHONPATH ./run.sh
```

*(หมายเหตุ: ทดสอบแล้ว ระบบสามารถทำงานผ่านจุดที่มีนำเข้า `numpy` ได้สมบูรณ์ 100% ด้วยคำสั่งนี้)*

---

## 🛠️ โครงสร้างเบื้องหลัง

- **SQL Database**: ระบบจะทำการดึงข้อมูลและบันทึกสัญญาณ (Signals) ของวันนี้เข้าสู่ฐานข้อมูล SQLite เป็นอันดับแรก (อาจใช้เวลาประมาณ 1-2 นาทีในขั้นตอนนี้)
- **RAG + LLM**: เมื่อโหลดข้อมูลสำเร็จ ระบบจะแสดงหน้าต่างให้ผู้ใช้ป้อนคำถาม โดยใช้ 2 Experts (DeepSeek, Qwen) และ 1 Master Judge (MiniMax) วิเคราะห์ข้อมูลจากฐานข้อมูลโดยอัตโนมัติ
- **Environment**: ตัวจัดการหลัก `run.sh` จะเรียกใช้ Python จาก Virtual Environment (`venv`) ของโปรเจกต์โดยตรง คุณจึงไม่จำเป็นต้องพิมพ์ `source venv/bin/activate` ก่อนรันระบบ
