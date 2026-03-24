# 📈 AI Stock Trend Predictor (S&P 500)

**AI Stock Trend Predictor** คือเว็บแอปพลิเคชันที่สร้างขึ้นจากกระบวนการ End-to-End Machine Learning Pipeline เพื่อพยากรณ์ทิศทางราคาหุ้นในกลุ่ม S&P 500 สำหรับวันพรุ่งนี้ (ขึ้น หรือ ลง) โดยอาศัยข้อมูลทางสถิติและตัวชี้วัดทางเทคนิค (Technical Indicators) 

โปรเจกต์นี้เป็นส่วนหนึ่งของการศึกษาด้าน Data Science เพื่อเปลี่ยนข้อมูลกราฟที่ซับซ้อนให้เป็นการตัดสินใจแบบ Data-Driven

## ✨ ฟีเจอร์หลัก (Key Features)

แอปพลิเคชันนี้รองรับการทำงาน 2 โหมดหลัก เพื่อตอบโจทย์ทั้งผู้ใช้งานทั่วไปและนักวิเคราะห์:

* **🤖 โหมดอัตโนมัติ (Auto-Fetch):** ดึงข้อมูลราคาหุ้นแบบ Real-time ผ่าน Yahoo Finance API คำนวณ Indicators อัตโนมัติ และแสดงผลคำทำนายพร้อมเปอร์เซ็นต์ความน่าจะเป็น (Probability) ในเวลาเพียงไม่กี่วินาที
* **✍️ โหมดจำลองสถานการณ์ (Manual Sandbox):** ห้องทดลองแบบ What-If Analysis ที่เปิดให้ผู้ใช้ปรับแต่งค่าตัวแปรต่างๆ (เช่น RSI, MACD, ความผันผวน) ด้วยตนเองผ่าน Slider เพื่อดูการตอบสนองและวิธีคิดของ AI 

## 🧠 สถาปัตยกรรมโมเดล (Model Architecture)

* **Algorithm:** Random Forest Classifier (Champion Model)
* **Dataset:** ข้อมูล Panel Data หุ้น 505 บริษัท ระยะเวลา 5 ปีย้อนหลัง (กว่า 600,000 รายการ)
* **Feature Engineering (10 มิติ):** * โมเมนตัมและแนวโน้ม: `RSI_14`, `MACD`, `SMA_7`, `EMA_14`
  * ผลตอบแทนและความเสี่ยง: `Log Return`, `Lag 1,3,7 Return`, `Volatility 7-Days`
  * ภาพรวมตลาด: `Relative Return` (เทียบกับดัชนี SPY)
* **Data Prep:** มีการจัดการ Outlier เชิงลึกด้วยวิธี IQR แยกตามรายบริษัท เพื่อลด Noise ในโมเดล

## 🛠️ เทคโนโลยีที่ใช้ (Tech Stack)

* **ภาษาโปรแกรม:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Retrieval:** yfinance
* **Web Deployment:** Streamlit
* **Visualization:** Matplotlib, Seaborn

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
├── app.py                      # โค้ดหลักของ Web Application (Streamlit)
├── 67160346_DS_S&PBACKTEST5YR.ipynb # Jupyter Notebook แสดง EDA, Feature Engineering และ Model Training
├── stock_champion_model.pkl    # ไฟล์โมเดล Random Forest ที่ผ่านการเทรน
├── stock_features.json         # ลิสต์ตัวแปรที่โมเดลต้องใช้
├── stock_metadata.json         # ข้อมูลสถิติของโมเดล (Accuracy, Name)
├── ticker_list.json            # รายชื่อตัวย่อหุ้น S&P 500 สำหรับสร้างเมนู
├── requirements.txt            # รายชื่อไลบรารีที่จำเป็นสำหรับการรันโปรเจกต์
└── README.md                   # ไฟล์อธิบายโปรเจกต์


🚀 วิธีการติดตั้งและทดลองรัน (Installation & Setup)
โคลน Repository:

Bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
ติดตั้งไลบรารีที่จำเป็น: (แนะนำให้ใช้ Virtual Environment)

Bash
pip install -r requirements.txt
รันแอปพลิเคชัน Streamlit:

Bash
streamlit run app.py
เปิดใช้งานหน้าเว็บ: เข้าไปที่เว็บบราวเซอร์ของคุณตาม URL ที่แสดงใน Terminal (โดยปกติคือ http://localhost:8501)