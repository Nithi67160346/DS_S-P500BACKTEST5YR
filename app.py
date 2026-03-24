import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import yfinance as yf
import os

# ==========================================
# 0. พจนานุกรมคำอธิบายตัวแปร (Feature Dictionary)
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

# แปลงชื่อเป็นพิมพ์เล็กทั้งหมดเพื่อการค้นหาที่แม่นยำ (Case-Insensitive)
feature_dict = {k.lower(): v for k, v in feature_dict_raw.items()}

# ==========================================
# 1. โหลดข้อมูลโมเดลและไฟล์ตั้งค่าต่างๆ (รองรับ Public URL)
# ==========================================
@st.cache_resource 
def load_assets():
    try:
        model = joblib.load('stock_champion_model.pkl')
        with open('stock_features.json', 'r') as f:
            features = json.load(f)
        with open('stock_metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open('ticker_list.json', 'r') as f:
            stock_list = json.load(f)
        return model, features, metadata, stock_list
    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาดในการโหลดไฟล์ระบบ: {e} \nกรุณาตรวจสอบว่ามีไฟล์ .pkl และ .json ครบถ้วน")
        return None, [], {}, []

model, features, metadata, stock_list = load_assets()

# ==========================================
# 2. ฟังก์ชันคำนวณ Features (AI ดึงข้อมูลอัตโนมัติ)
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
    df['Volatility_7'] = df['log_return'].rolling(window=7).std() 
    
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
# 3. ตั้งค่าหน้าเว็บหลักและ Sidebar UI
# ==========================================
st.set_page_config(page_title="AI Stock Predictor", page_icon="📈", layout="centered")

# เมนูด้านข้าง
st.sidebar.title("⚙️ เมนูการใช้งาน")
app_mode = st.sidebar.radio(
    "เลือกโหมดการพยากรณ์:",
    ["🤖 โหมดอัตโนมัติ (Auto-Fetch)", "✍️ โหมดจำลองสถานการณ์ (Manual)"]
)

st.sidebar.markdown("---")
if metadata:
    st.sidebar.success(f"🏆 **Model:** {metadata.get('model_name', 'Random Forest')}\n\n🎯 **Accuracy:** {metadata.get('test_accuracy', 0)*100:.2f}%")

# ส่วนอธิบาย Features ให้คนทั่วไปเข้าใจง่าย (UI สำหรับคนไม่รู้จัก ML)
with st.sidebar.expander("📚 อภิธานศัพท์ (ความหมายของตัวแปร)"):
    for k, v in feature_dict_raw.items():
        st.markdown(f"**{k.upper()}**:\n{v}")

# ==========================================
# 4. การทำงานแยกตามโหมด
# ==========================================

# ---------------------------------------------------------
# โหมด 1: โหมดอัตโนมัติ (Auto-Fetch)
# ---------------------------------------------------------
if app_mode == "🤖 โหมดอัตโนมัติ (Auto-Fetch)":
    st.title("🤖 AI Stock Trend Predictor")
    st.markdown("ระบบวิเคราะห์และพยากรณ์ทิศทางหุ้นอัจฉริยะ เพียงเลือกชื่อหุ้น AI จะดึงข้อมูลล่าสุดมาคำนวณให้ทันที!")
    st.markdown("---")
    
    if stock_list:
        default_index = stock_list.index("AAPL") if "AAPL" in stock_list else 0
        ticker = st.selectbox("🔍 เลือกหรือพิมพ์ตัวย่อหุ้น S&P 500 (เช่น AAPL, TSLA):", options=stock_list, index=default_index)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 ดึงข้อมูลและวิเคราะห์แนวโน้ม", use_container_width=True):
            with st.spinner(f'กำลังเชื่อมต่อข้อมูลตลาดแบบ Real-time ดึงข้อมูล {ticker}...'):
                result = calculate_features(ticker)
                
                if result is None:
                    st.error("❌ ไม่พบข้อมูลหุ้นนี้ หรือระบบเซิร์ฟเวอร์มีปัญหา โปรดลองใหม่อีกครั้ง")
                else:
                    input_df, current_price = result
                    
                    st.markdown("### 📊 ข้อมูล Indicator ล่าสุดที่ AI นำไปคำนวณ:")
                    st.dataframe(input_df.style.format("{:.4f}"))
                    
                    # พยากรณ์ผล
                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0]
                    
                    st.markdown("---")
                    st.markdown(f"### 🎯 สรุปผลการพยากรณ์แนวโน้ม **{ticker}** สำหรับวันพรุ่งนี้:")
                    st.info(f"💵 ราคาปิดล่าสุด: **${float(current_price):.2f}**")
                    
                    # แสดงผลแบบเข้าใจง่ายพร้อม Probability
                    if prediction == 1:
                        st.success(f"**สัญญาณ: ขาขึ้น (UP Trend) 🚀**")
                        st.write(f"โอกาสความน่าจะเป็น (Probability): **{probability[1]*100:.2f}%**")
                        st.progress(float(probability[1])) 
                    else:
                        st.error(f"**สัญญาณ: ขาลง (DOWN Trend) 📉**")
                        st.write(f"โอกาสความน่าจะเป็น (Probability): **{probability[0]*100:.2f}%**")
                        st.progress(float(probability[0]))

# ---------------------------------------------------------
# โหมด 2: โหมดจำลองสถานการณ์ (Manual Sandbox)
# ---------------------------------------------------------
elif app_mode == "✍️ โหมดจำลองสถานการณ์ (Manual)":
    st.title("✍️ โหมดจำลองสถานการณ์ (Sandbox)")
    st.markdown("ทดสอบสมมติฐานโดยปรับเปลี่ยนตัวเลข เพื่อดูว่า AI จะตอบสนองอย่างไร")
    st.markdown("---")
    
    # --- ส่วนที่ 1: ดึงข้อมูลตั้งต้น (Magic Pre-fill) ---
    st.markdown("### 🪄 1. ดึงข้อมูลตั้งต้น (จะได้ไม่ต้องกรอกเองทั้งหมด!)")
    
    col_t, col_b = st.columns([3, 1])
    with col_t:
        manual_ticker = st.selectbox("เลือกหุ้นเริ่มต้น:", options=stock_list, index=stock_list.index("AAPL") if "AAPL" in stock_list else 0)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("🔄 ดึงค่าล่าสุด", use_container_width=True)

    # สร้างหน่วยความจำ (Session State) เพื่อจำค่าที่กรอกไว้
    if 'manual_inputs' not in st.session_state:
        st.session_state.manual_inputs = {feat: 0.0 for feat in features}
        if 'RSI_14' in st.session_state.manual_inputs: st.session_state.manual_inputs['RSI_14'] = 50.0

    # ดึงข้อมูลอัตโนมัติเมื่อกดปุ่ม
    if fetch_btn:
        with st.spinner(f"กำลังดึงข้อมูล {manual_ticker} มาเป็นค่าเริ่มต้น..."):
            res = calculate_features(manual_ticker)
            if res is not None:
                input_df, _ = res
                for feat in features:
                    st.session_state.manual_inputs[feat] = float(input_df.iloc[0][feat])
                st.success(f"✅ ดึงข้อมูลสำเร็จ! ลองเลื่อนลงไปปรับแก้ตัวเลขด้านล่างดูได้เลย")
            else:
                st.error("❌ ดึงข้อมูลไม่สำเร็จ")

    # --- ส่วนที่ 2: ปรับแต่งตัวเลขพร้อม Validation ---
    st.markdown("### 🎛️ 2. ปรับแต่งตัวเลข (What-If Analysis)")
    
    # จัดกลุ่มตัวแปรให้เป็นระเบียบ
    tech_cols, return_cols, vol_cols, other_cols = [], [], [], []
    for f in features:
        fl = str(f).lower()
        if any(x in fl for x in ['rsi', 'macd', 'sma', 'ema']): tech_cols.append(f)
        elif 'return' in fl or 'lag' in fl: return_cols.append(f)
        elif 'vol' in fl: vol_cols.append(f)
        else: other_cols.append(f)

    # สร้าง Tabs เพื่อ UI ที่สะอาดตา
    tabs = st.tabs(["📈 อินดิเคเตอร์หลัก", "💰 ผลตอบแทน (Returns)", "🌪️ ความผันผวน (Risk)"] + (["📦 อื่นๆ"] if other_cols else []))
    
    user_input = {}

    # ฟังก์ชันช่วยสร้าง Input พร้อมระบบ Validation ป้องกันค่าที่ผิดปกติ
    def create_input(feat, col):
        search_key = str(feat).strip().lower()
        desc = feature_dict.get(search_key, f"ตัวแปร {feat}")
        val = st.session_state.manual_inputs.get(feat, 0.0)
        
        with col:
            if 'rsi' in search_key: 
                # RSI บังคับใช้แถบเลื่อน 0-100
                return st.slider(f"{feat} ❓", min_value=0.0, max_value=100.0, value=max(0.0, min(100.0, float(val))), step=1.0, help=desc + " (ค่า 0-100)")
            elif 'sma' in search_key or 'ema' in search_key: 
                # ราคาเฉลี่ยห้ามติดลบ
                return st.number_input(f"{feat} ❓", min_value=0.0, value=max(0.0, float(val)), step=1.0, format="%.4f", help=desc + " (ราคาห้ามติดลบ)")
            elif 'vol' in search_key:
                # ความผันผวนห้ามติดลบ
                return st.number_input(f"{feat} ❓", min_value=0.0, value=max(0.0, float(val)), step=0.005, format="%.4f", help=desc + " (ความผันผวนห้ามติดลบ)")
            else: 
                # ผลตอบแทนและ MACD กรอกติดลบได้
                return st.number_input(f"{feat} ❓", value=float(val), step=0.005, format="%.4f", help=desc)

    # วาดกล่องรับค่าลงใน Tab
    with tabs[0]:
        st.caption("กลุ่มตัวแปรเชิงเทคนิค ลองเลื่อนแถบ RSI ดูก็ได้ครับ!")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(tech_cols): user_input[feat] = create_input(feat, c1 if i % 2 == 0 else c2)
    with tabs[1]:
        st.caption("กลุ่มตัวแปรผลตอบแทน (สามารถติดลบได้ มักเป็นทศนิยมค่าน้อยๆ เช่น 0.0150)")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(return_cols): user_input[feat] = create_input(feat, c1 if i % 2 == 0 else c2)
    with tabs[2]:
        st.caption("กลุ่มตัวแปรความเสี่ยงและความผันผวน (ค่าต้องไม่ติดลบ)")
        c1, c2 = st.columns(2)
        for i, feat in enumerate(vol_cols): user_input[feat] = create_input(feat, c1 if i % 2 == 0 else c2)
    if other_cols:
        with tabs[3]:
            c1, c2 = st.columns(2)
            for i, feat in enumerate(other_cols): user_input[feat] = create_input(feat, c1 if i % 2 == 0 else c2)

    st.markdown("---")
    
    # --- ส่วนที่ 3: ปุ่มทำนายผล ---
    if st.button("🔮 ทำนายผลจากข้อมูลจำลอง", use_container_width=True):
        input_df = pd.DataFrame([user_input])
        
        # 🌟 สั่งให้เรียงคอลัมน์กลับไปเหมือนตอนที่ AI เรียนมาเป๊ะๆ (แก้ Error The feature names should match...)
        input_df = input_df[features] 
        
        with st.spinner('AI กำลังวิเคราะห์ข้อมูลจำลอง...'):
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
        
        st.markdown("### 🎯 ผลการพยากรณ์จำลอง:")
        
        if prediction == 1:
            st.success(f"**สัญญาณ: ขาขึ้น (UP Trend) 🚀**")
            st.write(f"โอกาสความน่าจะเป็น (Probability): **{probability[1]*100:.2f}%**")
            st.progress(float(probability[1])) 
        else:
            st.error(f"**สัญญาณ: ขาลง (DOWN Trend) 📉**")
            st.write(f"โอกาสความน่าจะเป็น (Probability): **{probability[0]*100:.2f}%**")
            st.progress(float(probability[0]))

# ==========================================
# 5. Disclaimer ตามเกณฑ์
# ==========================================
st.markdown("---")
st.warning(
    "⚠️ **ข้อจำกัดความรับผิดชอบ (Disclaimer):**\n"
    "แอปพลิเคชันนี้เป็นส่วนหนึ่งของโปรเจกต์ทางการศึกษาด้าน Data Science & Machine Learning เท่านั้น "
    "ผลการทำนายอ้างอิงจากแบบจำลองทางสถิติในอดีต ซึ่ง **ไม่สามารถรับประกันผลลัพธ์ในอนาคตได้** "
    "ผู้ใช้งานไม่ควรใช้ข้อมูลนี้เป็นคำแนะนำในการตัดสินใจซื้อขายหรือลงทุนทางการเงินใดๆ ทั้งสิ้น"
)