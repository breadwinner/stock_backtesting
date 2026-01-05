import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
from backtesting import Backtest, Strategy
import os
import streamlit.components.v1 as components

# ==========================================
# 1. 策略类定义 (直接复用你的核心逻辑)
# ==========================================
class TDSequentialStrategy(Strategy):
    tp_pct = 0.20 

    def init(self):
        close = self.data.Close.s
        low = self.data.Low.s
        
        # 计算 TD Setup (连跌4天)
        self.td_setup_count = self.I(self.BARSLASTCOUNT, close < pd.Series(close).shift(4))
        
        # 卖出端状态变量
        self.sell_setup_count = 0        
        self.sell_countdown_active = False 
        self.sell_countdown_count = 0    

        def get_vol_sma(volume_array, length=5):
            return ta.sma(pd.Series(volume_array), length=length).to_numpy()

        # 量能均线
        self.vol_ma5 = self.I(get_vol_sma, self.data.Volume, 5)

        self.current_structure_low = float('inf') 
        self.stop_loss_price = 0                  

    def BARSLASTCOUNT(self, condition):
        # 转换为 Series 确保 groupby 正常工作
        cond_series = pd.Series(condition, index=self.data.index)
        return cond_series.groupby((cond_series != cond_series.shift()).cumsum()).cumsum().to_numpy()

    def next(self):
        if len(self.data) < 15: # 稍微增加缓冲
            return
        
        # A. 数据准备
        curr_close = self.data.Close[-1]
        curr_open  = self.data.Open[-1]
        curr_high  = self.data.High[-1]
        curr_vol   = self.data.Volume[-1]
        # ma_vol     = self.vol_ma5[-1] # 有时可能会是 NaN，需要处理，但在 self.I 处理后通常安全
        
        # 安全获取 ma_vol
        try:
            ma_vol = self.vol_ma5[-1]
        except:
            ma_vol = 0

        ref_c_4 = self.data.Close[-5] 
        ref_h_2 = self.data.High[-3]  

        # B. 卖出结构计算
        if curr_close > ref_c_4:
            self.sell_setup_count += 1
        else:
            self.sell_setup_count = 0
            
        if self.sell_setup_count == 9:
            self.sell_countdown_active = True
            self.sell_countdown_count = 0

        if self.sell_countdown_active:
            if curr_close > ref_h_2:
                self.sell_countdown_count += 1

        # C. 信号定义
        signal_hj8 = (self.sell_setup_count == 9) or (self.sell_setup_count == 18)

        signal_hj38 = False
        if self.sell_setup_count == 9:
            prev_12_highs = np.max(self.data.High[-13:-1])
            if (curr_high < prev_12_highs) or (curr_close < curr_open):
                signal_hj38 = True

        signal_hj39 = (self.sell_countdown_count == 13)

        should_sell = signal_hj8 or signal_hj39
        
        # 修正索引访问，避免越界
        try:
            hj31_signal = (self.data.Close[-1] > self.data.High[-3]) and \
                          (self.data.Close[-2] <= self.data.High[-4])
            
            hj51_54_signal = (self.td_setup_count[-1] >= 13) and \
                             (self.data.Close[-1] > self.data.High[-2])
        except IndexError:
            return

        # 买入逻辑
        if not self.position:
            volume_confirmed = curr_vol > ma_vol
            if (self.td_setup_count[-1] >= 13) and (hj31_signal or hj51_54_signal):
                 self.buy()

        elif self.position:
            entry_price = self.trades[-1].entry_price
            stop_loss_price   = entry_price * 0.85 
            
            if self.data.Low[-1] <= stop_loss_price:
                self.position.close()
            elif should_sell:
                self.position.close()

# ==========================================
# 2. Streamlit 界面逻辑
# ==========================================
st.set_page_config(page_title="TD 策略回测平台", layout="wide")

st.title("📈 TD Sequential 量化回测平台")
st.markdown("基于 TD 序列 (9转/13转) 的自动化回测系统")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 参数设置")
    ticker = st.text_input("股票代码 (Yahoo Finance)", value="TSLA")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("结束日期", value=pd.to_datetime("today"))
        
    cash = st.number_input("初始资金", value=100000, step=10000)
    commission = st.number_input("交易佣金 (比例)", value=0.001, step=0.0001, format="%.4f")
    
    run_btn = st.button("🚀 开始回测", use_container_width=True)

# 缓存数据下载，避免重复请求
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        if len(data) == 0:
            return None
            
        # 处理 MultiIndex (Yahoo Finance 新版特性)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # 重命名为 Backtesting 库要求的格式
        data.columns = [c.capitalize() for c in data.columns]
        
        # 移除时区信息 (Backtesting 不喜欢 tz-aware index)
        data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        st.error(f"数据下载失败: {e}")
        return None

# 主执行逻辑
if run_btn:
    with st.spinner(f'正在下载 {ticker} 数据并进行计算...'):
        df = load_data(ticker, start_date, end_date)
        
        if df is not None and len(df) > 50:
            # 1. 运行回测
            bt = Backtest(df, TDSequentialStrategy, cash=cash, commission=commission)
            stats = bt.run()
            
            # 2. 显示关键指标
            st.subheader("📊 回测结果摘要")
            
            # 创建 4 列显示核心数据
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("总回报率 (Return)", f"{stats['Return [%]']:.2f}%")
            kpi2.metric("胜率 (Win Rate)", f"{stats['Win Rate [%]']:.2f}%")
            kpi3.metric("最大回撤 (Max Drawdown)", f"{stats['Max. Drawdown [%]']:.2f}%")
            kpi4.metric("夏普比率 (Sharpe Ratio)", f"{stats['Sharpe Ratio']:.2f}")

            # 3. 显示详细数据表格
            with st.expander("查看详细回测报告"):
                st.dataframe(stats.astype(str)) # 转字符串显示避免格式问题

            # 4. 显示交互式图表
            st.subheader("🕯️ 资金曲线与交易点位")
            
            # 技巧：Backtesting.py 默认生成 HTML 文件
            # 我们将其保存为临时文件，然后用 Streamlit 组件读取
            plot_file = "temp_plot.html"
            bt.plot(open_browser=False, filename=plot_file, resample=False)
            
            # 读取 HTML 并嵌入 Streamlit
            with open(plot_file, 'r', encoding='utf-8') as f:
                plot_html = f.read()
            
            # 使用 components.html 渲染，设置足够的高度
            components.html(plot_html, height=800, scrolling=True)
            
            # 清理临时文件 (可选)
            # os.remove(plot_file) 
            
        elif df is not None:
            st.warning("数据量不足，无法进行有效回测 (至少需要 50 个交易日)。")
        else:
            st.error("无法获取数据，请检查股票代码是否正确。")