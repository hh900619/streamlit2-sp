
#cd C:\Users\0619h\OneDrive\Desktop\streamlit2-sp
#git add .
#git commit -m "改"
#git push origin master
#8prUe$bf_._gFP5

import streamlit as st
st.set_page_config(page_title="比對", layout="wide")
import plotly.graph_objects as go
import time, random, datetime, os
import numpy as np
import pandas as pd
import yfinance as yf
from dtaidistance import dtw
from dotenv import load_dotenv
from supabase import create_client

import os
os.system('pip install plotly')

#######################################
# 1. 資料庫初始化
#######################################

load_dotenv('app2.env')

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    except Exception as e:
        st.error("❌ Supabase 設定錯誤，請檢查 .env 或 secrets.toml")
        st.stop()
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_stats_to_supabase(ticker, category, mode,
                           angle_min, angle_max, angle_avg,
                           total_score_min, total_score_max, total_score_avg):
    data = {
        "ticker": ticker,
        "category": category,  
        "mode": mode,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "angle_avg": angle_avg,
        "total_score_min": total_score_min,
        "total_score_max": total_score_max,
        "total_score_avg": total_score_avg,
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    response = supabase.table("stats").insert(data).execute()
    print("✅ 寫入 Supabase 完成", response)

def get_stat_ranges_from_supabase(ticker):
    response = supabase.table('stats').select(
        'angle_min, angle_max, angle_avg, total_score_min, total_score_max, total_score_avg'
    ).eq('ticker', ticker).execute()
    if not response.data:
        return None
    angle_min = float(min(float(item['angle_min']) for item in response.data))
    angle_max = float(max(float(item['angle_max']) for item in response.data))
    angle_avg = float(sum(float(item['angle_avg']) for item in response.data) / len(response.data))
    score_min = float(min(float(item['total_score_min']) for item in response.data))
    score_max = float(max(float(item['total_score_max']) for item in response.data))
    score_avg = float(sum(float(item['total_score_avg']) for item in response.data) / len(response.data))
    return {
        'angle_min': angle_min,
        'angle_max': angle_max,
        'angle_avg': angle_avg,
        'score_min': score_min,
        'score_max': score_max,
        'score_avg': score_avg
    }

def get_stats_count_from_supabase(ticker):
    try:
        response = supabase.table('stats').select('id').eq('ticker', ticker).execute()
        return len(response.data) if response.data else 0
    except Exception as e:
        st.warning(f"查詢 Supabase 失敗：{e}")
        return 0

#######################################
# 2. 資料下載與基本工具
#######################################
def download_data(ticker):
    df = yf.download(ticker, period='max', interval='1d', auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            df[col] = df["Close"]
    df = df[["Open", "High", "Low", "Close"]].dropna(how='any')
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df

def download_vix(start, end):
    vx = yf.download("^VIX", start=start, end=end, interval='1d', auto_adjust=False)
    if vx.empty:
        return pd.DataFrame()
    if isinstance(vx.columns, pd.MultiIndex):
        vx.columns = vx.columns.get_level_values(0)
    vx = vx[["Close"]].dropna(how='any')
    vx = vx[~vx.index.duplicated(keep='first')].sort_index()
    return vx

#######################################
# 3. 基本工具與指標計算
#######################################
def classify_ticker(ticker):
    """僅用於記錄資料庫中標的類型"""
    ticker = ticker.upper()
    if any(keyword in ticker for keyword in ['GOLD','SILVER','GC=F','SI=F','HG=F','PL=F','PA=F','XAU','XAG']):
        return "metal"
    elif any(keyword in ticker for keyword in ['CL=F','BRENT','NG=F','OIL','RB=F','HO=F','XLE']):
        return "energy"
    elif any(keyword in ticker for keyword in ['CORN','SOY','WHEAT','ZC=F','ZS=F','ZW=F','KC=F','COTTON','SUGAR']):
        return "agriculture"
    elif any(keyword in ticker for keyword in ['AAPL','MSFT','GOOG','GOOGL','META','TSLA','NVDA','AMD','INTC','AMZN','NFLX','BABA','CRM','ADBE','ORCL']):
        return "tech"
    elif any(keyword in ticker for keyword in ['SPY','QQQ','IWM','VOO','VTI','XLK','XLF','XLE','XLV','XLU','SMH','ARKK']):
        return "etf"
    elif any(keyword in ticker for keyword in ['ES=F','NQ=F','YM=F','RTY=F','SP500','NASDAQ','DOW','RUSSELL']):
        return "index_futures"
    elif any(keyword in ticker for keyword in ['JPY=X','EUR=X','AUD=X','GBP=X','USDCAD=X','USDCHF=X','USD','FX','EURUSD','GBPUSD']):
        return "forex"
    else:
        return "stock"

def weighted_random_choice(candidates):
    scores = np.array([s for (_, s) in candidates])
    weights = scores / scores.sum()
    idx = np.random.choice(len(candidates), p=weights)
    return candidates[idx]

def compute_average_price(df):
    return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

def compute_angle_series(df):
    avg = compute_average_price(df)
    diff = avg.diff().dropna()
    angles = np.degrees(np.arctan(diff))
    return angles.values  # 返回一維陣列

def compute_angle_similarity(current_angles, candidate_angles, method="absolute"):
    if method == "absolute":
        diff = np.abs(current_angles - candidate_angles)
        mean_diff = np.mean(diff)
        score = max(0, 100 - (mean_diff / 180) * 100)
    elif method == "dtw":
        dtw_distance = dtw.distance(current_angles, candidate_angles)
        # 以每個點的平均差異來正規化 (理論最大差異約 180 度)
        normalized_distance = dtw_distance / len(current_angles)
        score = max(0, 100 - (normalized_distance / 180) * 100)
    else:
        score = 0
    return score

def compute_atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = np.abs(df["High"] - df["Close"].shift(1))
    lc = np.abs(df["Low"] - df["Close"].shift(1))
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_atr_similarity(current_atr, candidate_atr):
    diff = np.abs(current_atr - candidate_atr)
    rel_diff = diff / (current_atr + 1e-9)
    score = max(0, 100 - rel_diff * 100)
    return score

def compute_vix_similarity(current_vix, candidate_vix):
    diff = abs(current_vix - candidate_vix)
    rel_diff = diff / (current_vix + 1e-9)
    score = max(0, 100 - rel_diff * 100)
    return score

# 針對利率與通膨，由用戶輸入候選與當前值，這裡直接計算差距百分比
def compute_factor_similarity(current_val, candidate_val):
    diff = abs(current_val - candidate_val)
    score = max(0, 100 - (diff / 10) * 100)
    return score

#######################################
# 4. 新角度比對法的指標計算
#######################################
def compare_segments(current_df, candidate_df, extra_factors):
    """
    比對兩個區段：
      - 利用角度比對計算 angle_similarity
      - 如 extra_factors 為字典，包含 ATR, VIX, Interest, Inflation 的分數（預先計算好）
      - 回傳加權總分
    """
    current_angles = compute_angle_series(current_df)[-10:]  # 取最後 10 根角度
    candidate_angles = compute_angle_series(candidate_df)[-10:]
    angle_score = compute_angle_similarity(current_angles, candidate_angles)
    total = 0
    total += 0.5 * angle_score  # 角度比對佔 50%
    if "ATR" in extra_factors:
        total += 0.15 * extra_factors["ATR"]
    if "VIX" in extra_factors:
        total += 0.15 * extra_factors["VIX"]
    if "Interest" in extra_factors:
        total += 0.1 * extra_factors["Interest"]
    if "Inflation" in extra_factors:
        total += 0.1 * extra_factors["Inflation"]
    return total

def find_best_match_angle(df, vix_df, seg_len, fut_len, total_score_thr, topN, ticker, category, mode, extra_flags, extra_values, use_dtw=False):
    """
    遍歷歷史資料，找出與目前區段角度序列及額外指標最相似的候選區段
    use_dtw: 若為 True，則使用 DTW 計算角度相似度；否則採用絕對平均差法
    """
    if len(df) < seg_len + fut_len:
        return None
    current_segment = df.iloc[-seg_len:]
    candidates = []
    # 計算當前區段的 ATR 與 VIX（若啟用）
    current_atr = None
    if extra_flags.get("ATR", False):
        atr_series = compute_atr(df)
        current_atr = atr_series.iloc[-seg_len:].mean()
    current_vix = None
    if extra_flags.get("VIX", False):
        current_vix = vix_df.loc[vix_df.index >= current_segment.index[0]]["Close"].mean()

    # 遍歷歷史資料中每個可能候選區段
    for i in range(seg_len, len(df) - fut_len):
        candidate_segment = df.iloc[i - seg_len:i]
        # 計算角度比對得分，根據 use_dtw 決定計算方法
        method = "dtw" if use_dtw else "absolute"
        angle_score = compute_angle_similarity(
            compute_angle_series(current_segment)[-10:], 
            compute_angle_series(candidate_segment)[-10:],
            method=method
        )
        extra_scores = {}
        if extra_flags.get("ATR", False) and current_atr is not None:
            candidate_atr = compute_atr(df).iloc[i - seg_len:i].mean()
            extra_scores["ATR"] = compute_atr_similarity(current_atr, candidate_atr)
        if extra_flags.get("VIX", False) and current_vix is not None:
            candidate_vix = vix_df.loc[vix_df.index >= candidate_segment.index[0]]["Close"].mean()
            extra_scores["VIX"] = compute_vix_similarity(current_vix, candidate_vix)
        if extra_flags.get("Interest", False):
            extra_scores["Interest"] = compute_factor_similarity(extra_values.get("Interest_current", 0),
                                                                  extra_values.get("Interest_candidate", 0))
        if extra_flags.get("Inflation", False):
            extra_scores["Inflation"] = compute_factor_similarity(extra_values.get("Inflation_current", 0),
                                                                   extra_values.get("Inflation_candidate", 0))
        # 綜合各指標得分
        final_score = 0.5 * angle_score
        if "ATR" in extra_scores:
            final_score += 0.15 * extra_scores["ATR"]
        if "VIX" in extra_scores:
            final_score += 0.15 * extra_scores["VIX"]
        if "Interest" in extra_scores:
            final_score += 0.1 * extra_scores["Interest"]
        if "Inflation" in extra_scores:
            final_score += 0.1 * extra_scores["Inflation"]
        if final_score >= total_score_thr:
            candidates.append((i, final_score))
    final_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:topN]
    if candidates:
        scores = [s for (_, s) in candidates]
        angle_min, angle_max, angle_avg = min(scores), max(scores), np.mean(scores)
    else:
        angle_min = angle_max = angle_avg = 0
    save_stats_to_supabase(
        ticker=ticker,
        category=category,
        mode=mode,
        angle_min=angle_min,
        angle_max=angle_max,
        angle_avg=angle_avg,
        total_score_min=total_score_thr,
        total_score_max=max(scores) if candidates else 0,
        total_score_avg=np.mean(scores) if candidates else 0
    )
    return final_candidates

#######################################
# 5. 預設參數生成
#######################################
def get_default_params_v2(mode, angle_range, score_range):
    """
    根據歷史統計區間與模式設定預設參數
    回傳 (total_score_thr, topN)
    """
    angle_min, angle_max = angle_range
    score_min, score_max = score_range
    if mode == "保守":
        total_score_thr = int((score_min + score_max) / 3)  # 較低門檻
        topN = 30
    elif mode == "平衡":
        total_score_thr = int((score_min + score_max) / 2)
        topN = 50
    elif mode == "寬鬆":
        total_score_thr = int(score_min * 0.8)
        topN = 80
    else:  # 自訂模式
        total_score_thr = int(st.slider("總分門檻 (0~100)", min_value=20, max_value=100, value=80, step=1))
        topN = int(st.slider("TopN 隨機選擇", 1, 200, 50))
    return total_score_thr, topN

def copy_future_bars_percent_mode(df, best_i, fut_len):
    """
    複製未來 K 棒，採用百分比模式進行計算
    """
    samp = df.iloc[best_i: best_i + fut_len].copy()
    if len(samp) < fut_len:
        return None, None
    last_close = df["Close"].iloc[-1]
    hist_prev_close = df["Close"].iloc[best_i - 1]
    new_ohlc = []
    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=fut_len,
        freq='B'
    )
    for i in range(fut_len):
        row = samp.iloc[i]
        ratio_open = row["Open"] / max(abs(hist_prev_close), 1e-9)
        new_open = last_close * ratio_open
        ratio_h = row["High"] / max(abs(row["Open"]), 1e-9)
        ratio_l = row["Low"] / max(abs(row["Open"]), 1e-9)
        ratio_c = row["Close"] / max(abs(row["Open"]), 1e-9)
        new_high = new_open * ratio_h
        new_low = new_open * ratio_l
        new_close = new_open * ratio_c
        hi = max(new_open, new_high, new_low, new_close)
        lo = min(new_open, new_high, new_low, new_close)
        new_ohlc.append([new_open, hi, lo, new_close])
        last_close = new_close
        hist_prev_close = row["Close"]
    return future_dates, new_ohlc

#######################################
# 6. 主程式 main()
#######################################
def main():
    st.title("比對 - 角度比對新方法")
    ticker = st.text_input("股票代號 (例如 AAPL):", value="AAPL")
    seg_len = st.number_input("Segment Length (計算區段長度)", 5, 50, 10)
    fut_len = st.number_input("Future Copy (預測未來 K 棒數)", 1, 20, 5)
    total_predict = st.slider("總預測天數", 5, 200, 50)
    
    # 額外指標設定
    st.subheader("選擇額外指標（可選）")
    include_atr = st.checkbox("包含 ATR 指標", value=True)
    include_vix = st.checkbox("包含 VIX 指標", value=True)
    include_interest = st.checkbox("包含 利率 指標", value=False)
    include_inflation = st.checkbox("包含 通膨 指標", value=False)
    extra_flags = {
        "ATR": include_atr,
        "VIX": include_vix,
        "Interest": include_interest,
        "Inflation": include_inflation
    }
    extra_values = {}
    if include_interest:
        extra_values["Interest_current"] = st.number_input("目前利率 (%)", value=3.0)
        extra_values["Interest_candidate"] = st.number_input("候選利率 (%)", value=3.0)
    if include_inflation:
        extra_values["Inflation_current"] = st.number_input("目前通膨率 (%)", value=2.0)
        extra_values["Inflation_candidate"] = st.number_input("候選通膨率 (%)", value=2.0)
    
    # 新增選項：是否使用 DTW 比對角度
    use_dtw = st.checkbox("使用 DTW 比對角度", value=True)
    
    # 取得產品類型（僅記錄用途）
    category = classify_ticker(ticker)
    
    # 讀取歷史統計資料
    stats_count = get_stats_count_from_supabase(ticker)
    stats = get_stat_ranges_from_supabase(ticker)
    if stats and stats_count >= 30:
        st.success(f"✅ 已累積 {stats_count} 筆歷史統計資料，啟用【歷史參數模式】")
        angle_min, angle_max = stats['angle_min'], stats['angle_max']
        score_min, score_max = stats['score_min'], stats['score_max']
        st.write(f"🌀 資料庫角度統計: min={angle_min:.2f}, max={angle_max:.2f}")
        st.write(f"📏 資料庫 Score 統計: min={score_min:.2f}, max={score_max:.2f}")
        angle_range = (max(angle_min * 0.9, 0), min(angle_max * 1.1, 100))
        score_range = (max(score_min * 0.9, 10), min(score_max * 1.1, 100))
        st.write(f"📈 建議角度區間：{angle_range}")
        st.write(f"📈 建議 Score 區間：{score_range}")
    else:
        st.warning(f"⚠️ 歷史資料僅 {stats_count} 筆，使用【預設參數模式】")
        angle_range = (30, 70)
        score_range = (30, 60)
        st.write(f"📈 預設角度區間：{angle_range}")
        st.write(f"📈 預設 Score 區間：{score_range}")
    
    mode = st.selectbox("⚙️ 選擇模式", ["保守", "平衡", "寬鬆", "自訂"])
    total_score_thr, topN = get_default_params_v2(mode, angle_range, score_range)
    st.success(f"🎲 產生參數：total_score_thr={total_score_thr}, topN={topN}")
    
    st.session_state['ticker'] = ticker
    st.session_state['category'] = category
    st.session_state['mode'] = mode
    st.session_state['topN'] = topN
    st.session_state['total_score_thr'] = total_score_thr

    if st.button("下載資料"):
        df = download_data(ticker)
        if df.empty:
            st.error(f"無法取得 {ticker} 資料")
            return
        vix_df = download_vix(df.index.min(), df.index.max())
        st.session_state['df'] = df
        st.session_state['vix_df'] = vix_df
        st.session_state['min_date'] = df.index.min().date()
        st.session_state['max_date'] = df.index.max().date()
        st.success(f"✅ 資料期間：{st.session_state['min_date']} ～ {st.session_state['max_date']}")

    if 'df' in st.session_state:
        st.write(f"資料期間：{st.session_state['min_date']} ～ {st.session_state['max_date']}")
        user_start_date = st.date_input("顯示起始日 (必選):", value=st.session_state['min_date'],
                                        min_value=st.session_state['min_date'],
                                        max_value=st.session_state['max_date'])
        use_custom_line = st.checkbox("是否加入自定垂直線？")
        user_line_date = None
        if use_custom_line:
            user_line_date = st.date_input("垂直線日期（可選）:",
                                           value=st.session_state['min_date'],
                                           min_value=st.session_state['min_date'],
                                           max_value=st.session_state['max_date'])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🛑 停止運行"):
                st.session_state['stop'] = True
                st.warning("⚠️ 預測已停止，請重新操作")
        with col2:
            if st.button("🔄 重新輸入 / 清除"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        if 'stop' not in st.session_state:
            st.session_state['stop'] = False

        if st.button("執行預測"):
            st.write(f"✅ 參數確認 - total_score_thr: {total_score_thr}, TopN: {topN}")
            df, vix_df = st.session_state['df'], st.session_state['vix_df']
            pred_days, loop_count, logs = 0, 0, []
            start_time = time.time()
            st.session_state['total_score_thr'] = total_score_thr
            st.session_state['topN'] = topN
            st.session_state['seg_len'] = seg_len
            st.session_state['fut_len'] = fut_len
            st.session_state['total_predict'] = total_predict
            st.session_state['stop'] = False

            last_real_date = df.index[-1]
            final_list = []

            while pred_days < st.session_state['total_predict'] and not st.session_state['stop']:
                loop_count += 1
                st.write(f"🔄 迴圈 {loop_count}，預測進度：{pred_days}/{total_predict} 天")
                candidates = find_best_match_angle(df, vix_df, seg_len, fut_len, total_score_thr, topN,
                                                   ticker, category, mode, extra_flags, extra_values, use_dtw=use_dtw)
                if not candidates or len(candidates) == 0:
                    logs.append(f"第 {loop_count} 次 => 找不到符合條件樣本，停止")
                    break
                topN_dynamic = random.randint(
                    max(3, int(st.session_state['topN'] * 0.8)),
                    int(st.session_state['topN'] * 1.2)
                )
                st.info(f"🎲 動態 TopN = {topN_dynamic}")
                final_list = [(idx, score + random.uniform(-5, 5)) for idx, score in candidates]
                final_list = sorted(final_list, key=lambda x: x[1], reverse=True)
                if len(final_list) > topN_dynamic:
                    final_list = final_list[:topN_dynamic]
                st.success(f"✅ 最後篩選樣本數：{len(final_list)} (動態TopN: {topN_dynamic})")
                scores = [s for (_, s) in final_list]
                st.write(f"分數範圍：min={min(scores):.2f}, max={max(scores):.2f}, avg={np.mean(scores):.2f}")
                best_i, best_score = weighted_random_choice(final_list)
                logs.append(f"第 {loop_count} 次選中 index={best_i}, 加權分數={best_score:.2f}")
                dates, new_ohlc = copy_future_bars_percent_mode(df, best_i, fut_len)
                if dates is None:
                    logs.append("⚠ 後續K棒不足，停止")
                    break
                new_rows = pd.DataFrame(new_ohlc, columns=["Open", "High", "Low", "Close"], index=dates)
                df = pd.concat([df, new_rows]).sort_index()
                pred_days += fut_len
                elapsed = time.time() - start_time
                st.info(f"⏳ 目前總耗時：{elapsed:.2f} 秒")

            total_time = time.time() - start_time
            st.success(f"✅ 預測完成！總耗時 {total_time:.2f} 秒")
            fdf = df.copy()
            fdf = fdf.loc[fdf.index >= pd.to_datetime(user_start_date)]
            fig = go.Figure()
            fig.update_layout(
                hovermode='x unified',
                dragmode='zoom',
                xaxis=dict(
                    tickformat="%Y-%m-%d",
                    showgrid=True,
                    rangeslider_visible=False
                ),
                yaxis=dict(
                    fixedrange=False
                ),
                font=dict(family="Microsoft JhengHei", size=14),
            )
            fig.add_trace(go.Candlestick(
                x=fdf.index,
                open=fdf["Open"], high=fdf["High"],
                low=fdf["Low"], close=fdf["Close"],
                increasing_line_color="red",
                decreasing_line_color="green",
                name="歷史＋預測"
            ))
            fig.add_shape(
                type="line",
                x0=last_real_date,
                x1=last_real_date,
                y0=0,
                y1=1,
                line=dict(color="blue", width=2, dash="dash"),
                xref='x',
                yref='paper'
            )
            if user_line_date:
                fig.add_vline(
                    x=pd.to_datetime(user_line_date),
                    line_width=2, line_dash="dot", line_color="yellow",
                    annotation_text="自定日期", annotation_position="bottom right"
                )
            fig.update_xaxes(
                type='date',
                tickformat="%Y-%m-%d",
                showgrid=True,
                rangeslider_visible=False
            )
            if st.button("🔄 一鍵重置圖表"):
                st.rerun()
            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'doubleClick': 'reset',
                'editable': True,
                'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                'modeBarButtonsToAdd': ['resetScale2d', 'drawline', 'drawopenpath', 'drawcircle', 'eraseshape'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'prediction_chart',
                    'height': 600,
                    'width': 1200,
                    'scale': 2
                }
            })
            st.success("✅ 預測完成! 可滑鼠拖曳、縮放，點『🔄 一鍵重置圖表』恢復原視角")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
