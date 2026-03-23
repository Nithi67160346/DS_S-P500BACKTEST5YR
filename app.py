import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf

# ==========================================
# 0. พจนานุกรมคำอธิบายตัวแปร (แบบเจาะลึกสุดๆ)
# ==========================================
feature_dict_raw = {
    'log_return': "ผลตอบแทนปัจจุบัน (Log): วัดการเปลี่ยนแปลงราคาวันล่าสุด (ตัวเลขที่สะท้อนทิศทางทันที)",
    'lag_1_return': "ผลตอบแทนเมื่อวาน (Lag 1): AI ใช้จับโมเมนตัมระยะสั้น ว่าหุ้นมีแนวโน้มจะวิ่งไปทิศทางเดิมต่อหรือไม่",
    'lag_3_return': "ผลตอบแทนย้อนหลัง 3 วัน (Lag 3): ดูแนวโน้มย่อยในช่วงครึ่งสัปดาห์",
    'lag_7_return': "ผลตอบแทนสัปดาห์ที่แล้ว (Lag 7): AI ใช้หาจังหวะการเด้งกลับ (Rebound) หรือรอบการพักตัว",
    'stock_historical_vol': "ความผันผวนของหุ้น (Risk): ยิ่งค่านี้สูงแปลว่าราคาสวิงแรง (เสี่ยงมาก) AI จะระมัดระวังในการประเมิน",
    'volatility_7': "ความผันผวนของหุ้นระยะ 7 วัน (Risk): ดูว่ารอบสัปดาห์ที่ผ่านมาหุ้นแกว่งตัวรุนแรงแค่ไหน",
    'market_return': "ผลตอบแทนตลาด (SPY): บรรยากาศตลาดวันนั้น หากเป็นบวกแรงแปลว่าเงินกำลังไหลเข้าตลาดหุ้น",
    'relative_return': "ความแข็งแกร่งเทียบตลาด (Alpha): หุ้นที่ค่านี้บวกในวันที่ตลาดลบ AI จะมองว่าเป็นหุ้นที่แกร่งมาก",
    'sma_7': "ราคาเฉลี่ย 7 วัน (SMA 7): ถ้าราคาหุ้นยืนเหนือค่านี้ได้ AI จะมองว่าเทรนด์ระยะสั้นยังเป็นขาขึ้น",
    'ema_14': "ราคาเฉลี่ยถ่วงน้ำหนัก 14 วัน (EMA 14): ให้ความสำคัญกับราคาล่าสุดมากกว่าอดีต ใช้ดูเทรนด์ระยะกลาง",
    'rsi_14': "ดัชนี RSI 14 วัน: >70 คือคนแย่งซื้อจนแพง (ระวังหุ้นตก), <30 คือคนเทขายจนของถูก (เตรียมเด้งขึ้น)",
    'macd': "ดัชนี MACD: หากค่าเป็นบวกแปลว่าแรงส่งขาขึ้นกำลังมาแรง หากเป็นลบคือแรงเทขายกำลังคุมตลาด"
}

# 🌟 ฟิลเตอร์แปลงชื่อให้เป็นตัวพิมพ์เล็กทั้งหมดเพื่อป้องกันหาไม่เจอ
feature_dict = {k.lower(): v for k, v in feature_dict_raw.items()}

# ==========================================
# 1. โหลดข้อมูลโมเดลและไฟล์ตั้งค่าต่างๆ
# ==========================================
@st.cache_resource 
def load_assets():
    model = joblib.load('stock_champion_model.pkl')
    with open('stock_features.json', 'r') as f:
        features = json.load(f)
    with open('stock_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open('ticker_list.json', 'r') as f:
        stock_list = json.load(f)
    return model, features, metadata, stock_list

model, features, metadata, stock_list = load_assets()

# ==========================================
# 2. ฟังก์ชันคำนวณ Features (สำหรับโหมด Auto)
# ==========================================
def calculate_features(ticker_symbol):
    stock_data = yf.download(ticker_symbol, period="60d", progress=False)
    market_data = yf.download("SPY", period="60d", progress=False)
    
    if stock_data.empty or market_data.empty:
        return None
        
    def get_clean_close(data, ticker_name):
        if isinstance(data.columns, pd.MultiIndex):
            s = data['Close'][ticker_name]
        else:
            s = data['Close']
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s
        
    s_close = get_clean_close(stock_data, ticker_symbol)
    m_close = get_clean_close(market_data, "SPY")
    
    df = pd.DataFrame({'close': s_close, 'market_close': m_close})
    df.ffill(inplace=True)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['lag_1_return'] = df['log_return'].shift(1)
    df['lag_3_return'] = df['log_return'].shift(3)
    df['lag_7_return'] = df['log_return'].shift(7)
    
    df['Stock_Historical_Vol'] = df['log_return'].rolling(window=7).std()
    df['Volatility_7'] = df['log_return'].rolling(window=7).std() # เผื่อไว้ทั้ง 2 ชื่อ
    
    df['Market_return'] = np.log(df['market_close'] / df['market_close'].shift(1))
    df['Relative_Return'] = df['log_return'] - df['Market_return']
    
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    
    latest_data = df.iloc[-1]
    computed_lower = {str(k).lower(): v for k, v in latest_data.items()}
    
    input_dict = {}
    for feature in features:
        f_lower = str(feature).lower()
        if f_lower in computed_lower:
            val = computed_lower[f_lower]
            input_dict[feature] = 0.0 if pd.isna(val) else float(val)
        else:
            input_dict[feature] = 0.0 
        
    return pd.DataFrame([input_dict]), latest_data['close']

# ==========================================
# 3. ตั้งค่าหน้าเว็บหลักและเมนูด้านข้าง (Sidebar)
# ==========================================
st.set_page_config(page_title="Stock AI Predictor", page_icon="📈", layout="centered")

st.sidebar.title("⚙️ เมนูการใช้งาน")
app_mode = st.sidebar.radio(
    "เลือกโหมดการพยากรณ์:",
    ["🤖 โหมดอัตโนมัติ (Auto-Fetch)", "✍️ โหมดกรอกข้อมูลเอง (Manual)"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"🏆 **Model:** {metadata['model_name']}\n\n🎯 **Accuracy:** {metadata['test_accuracy']*100:.2f}%")

# ==========================================
# 4. หน้าจอการทำงาน
# ==========================================

# ---------------------------------------------------------
# โหมดอัตโนมัติ (Auto-Fetch)
# ---------------------------------------------------------
if app_mode == "🤖 โหมดอัตโนมัติ (Auto-Fetch)":
    st.title("🤖 AI Stock Trend Predictor")
    st.markdown("เพียงเลือกชื่อหุ้น AI จะดึงข้อมูลล่าสุดมาคำนวณและทำนายผลให้ทันที!")
    st.markdown("---")
    
    default_index = stock_list.index("AAPL") if "AAPL" in stock_list else 0
    ticker = st.selectbox("🔍 เลือกหรือพิมพ์ตัวย่อหุ้น (S&P 500):", options=stock_list, index=default_index)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 ดึงข้อมูลและวิเคราะห์แนวโน้ม", use_container_width=True):
        with st.spinner(f'กำลังเชื่อมต่อ Yahoo Finance ดึงข้อมูล {ticker}...'):
            result = calculate_features(ticker)
            
            if result is None:
                st.error("❌ ไม่พบข้อมูลหุ้นนี้ หรือเซิร์ฟเวอร์มีปัญหา โปรดลองใหม่อีกครั้ง")
            else:
                input_df, current_price = result
                
                st.markdown("### 📊 ข้อมูล Indicator ล่าสุดที่ AI คำนวณได้:")
                st.dataframe(input_df.style.format("{:.4f}"))
                
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]
                
                st.markdown("---")
                st.markdown(f"### 🎯 สรุปผลการพยากรณ์ของ **{ticker}** สำหรับพรุ่งนี้:")
                st.info(f"💵 ราคาปิดล่าสุด: **${float(current_price):.2f}**")
                
                if prediction == 1:
                    st.success(f"**สัญญาณ: ขาขึ้น (UP Trend) 🚀**")
                    st.write(f"ความมั่นใจของ AI: **{probability[1]*100:.2f}%**")
                    st.progress(float(probability[1])) 
                else:
                    st.error(f"**สัญญาณ: ขาลง (DOWN Trend) 📉**")
                    st.write(f"ความมั่นใจของ AI: **{probability[0]*100:.2f}%**")
                    st.progress(float(probability[0]))

# ---------------------------------------------------------
# โหมดกรอกข้อมูลเอง (Manual)
# ---------------------------------------------------------
elif app_mode == "✍️ โหมดกรอกข้อมูลเอง (Manual)":
    st.title("✍️ โหมดทดสอบกรอกข้อมูลเอง")
    st.markdown("ทดสอบการทำงานของโมเดล โดยผู้ใช้สามารถกำหนดค่า Indicator ต่างๆ ได้เอง")
    st.caption("💡 ทริค: นำเมาส์ไปชี้ที่เครื่องหมาย ❓ หลังชื่อตัวแปรเพื่อดูความหมายและวิธีที่ AI ตีความ")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    user_input = {}

    for i, feature in enumerate(features):
        # 🌟 ตัวค้นหาที่ฉลาดขึ้น: ตัดช่องว่าง และแปลงเป็นตัวพิมพ์เล็กทั้งหมดก่อนไปค้นใน Dictionary
        search_key = str(feature).strip().lower()
        desc = feature_dict.get(search_key, f"ค่าสถิติของตัวแปร {feature}")
        
        if i % 2 == 0:
            with col1:
                user_input[feature] = st.number_input(f"{feature}", value=0.0000, format="%.4f", help=desc)
        else:
            with col2:
                user_input[feature] = st.number_input(f"{feature}", value=0.0000, format="%.4f", help=desc)

    st.markdown("---")
    
    if st.button("🔮 ทำนายผลจากข้อมูลที่กรอก", use_container_width=True):
        input_df = pd.DataFrame([user_input])
        
        with st.spinner('AI กำลังวิเคราะห์สัญญาณ...'):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
        
        st.markdown("### 🎯 ผลการพยากรณ์:")
        
        if prediction == 1:
            st.success(f"**สัญญาณ: ขาขึ้น (UP Trend) 🚀**")
            st.write(f"ความมั่นใจของ AI: **{probability[1]*100:.2f}%**")
            st.progress(float(probability[1])) 
        else:
            st.error(f"**สัญญาณ: ขาลง (DOWN Trend) 📉**")
            st.write(f"ความมั่นใจของ AI: **{probability[0]*100:.2f}%**")
            st.progress(float(probability[0]))

st.markdown("<br><br><center><small>⚠️ โปรแกรมนี้เป็นผลงานโปรเจกต์ทางการศึกษา ไม่ใช่คำแนะนำทางการเงิน</small></center>", unsafe_allow_html=True)