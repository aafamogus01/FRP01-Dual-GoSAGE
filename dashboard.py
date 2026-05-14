import streamlit as st
import pandas as pd
import plotly.express as px
import time
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# ==========================================
# 1. KONFIGURASI TEMA (WAJIB PALING ATAS)
# ==========================================
st.set_page_config(page_title="Dual-GoSAGE Intelligence Radar", layout="wide")

# ==========================================
# 2. SISTEM KEAMANAN (PASSWORD LOGIN)
# ==========================================
def check_password():
    def password_entered():
        # Ganti "satriadata2026" dengan password rahasia buatanmu jika mau
        if st.session_state["password"] == "TENGKURAP":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Hapus dari memori demi keamanan
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<h2 style='text-align: center; color: #d173ff; margin-top: 50px;'>🔐 OTORISASI KEAMANAN DIPERLUKAN</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #e6d9f2;'>Sistem Intelijen FRP01-Dual-GoSAGE dikunci.</p>", unsafe_allow_html=True)
        # Bikin kolom ke tengah biar elegan
        _, col_login, _ = st.columns([1, 2, 1])
        with col_login:
            st.text_input("Masukkan Kode Akses Juri:", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.markdown("<h2 style='text-align: center; color: #d173ff; margin-top: 50px;'>🔐 OTORISASI KEAMANAN DIPERLUKAN</h2>", unsafe_allow_html=True)
        _, col_login, _ = st.columns([1, 2, 1])
        with col_login:
            st.text_input("Masukkan Kode Akses Juri:", type="password", on_change=password_entered, key="password")
            st.error("❌ Akses Ditolak. Kode tidak valid atau Anda tidak memiliki izin operasi.")
        return False
    return True

# ==========================================
# 3. JALANKAN DASHBOARD HANYA JIKA LOGIN BERHASIL
# ==========================================
if check_password():

    st.markdown("""
    <style>
        .stApp { background-color: #0a0410; color: #e6d9f2; }
        h1, h2, h3 { color: #d173ff; text-shadow: 0px 0px 15px #a600ff; font-family: 'Courier New', monospace; }
        .stMetric { background-color: #1a0b2e; border: 1px solid #7a1cbf; padding: 15px; border-radius: 8px; }
        div[data-testid="stMetricValue"] { color: #ff5eff; }
        .stDownloadButton button { background-color: #7a1cbf; color: white; width: 100%; border: 1px solid #d173ff; border-radius: 5px; font-weight: bold; }
        .stDownloadButton button:hover { background-color: #a600ff; border: 1px solid #ffffff; }
    </style>
    """, unsafe_allow_html=True)

    st.title("FRP01-Dual-GoSAGE: Algoritma Deep Learning Dual Layer Berbasis Dual GoSAGE untuk Deteksi Kolusi Tender")
    st.markdown("---")

    @st.cache_data(ttl=2) 
    def load_and_translate_data():
        try:
            df_vonis = pd.read_csv('Vonis_AI_DualGoSAGE.csv')
            df_eco = pd.read_csv('1_Data_Perusahaan_Ekonometrik.csv')
            df_transac = pd.read_csv('3_Relasi_Transaksional.csv')
            df_it = pd.read_csv('4_Relasi_Infrastruktur_IT.csv')

            # Anti-bentrok
            if 'Status_Aktual' in df_eco.columns:
                df_eco = df_eco.drop(columns=['Status_Aktual'])

            # Agregasi Anakan
            transac_agg = df_transac.groupby('ID_Perusahaan').agg({
                'Spread_SPD_Scaled': 'mean',
                'Relative_Diff_RD_Scaled': 'mean'
            }).reset_index()
            
            it_agg = df_it.groupby('ID_Perusahaan_A').agg({
                'Time_Gap_Bidding_Scaled': 'mean'
            }).reset_index().rename(columns={'ID_Perusahaan_A': 'ID_Perusahaan'})

            master = pd.merge(df_vonis, df_eco, on='ID_Perusahaan')
            master = pd.merge(master, transac_agg, on='ID_Perusahaan', how='left')
            master = pd.merge(master, it_agg, on='ID_Perusahaan', how='left').fillna(0)

            # Translasi Data
            master['Usia (Tahun)'] = np.round(np.abs((master['Usia_Scaled'] * 3.4) + 12)).astype(int)
            master['Win Rate (%)'] = np.round(np.clip((master['WinRate_Scaled'] * 20) + 30, 0, 100), 1)
            master['Modal (Miliar Rp)'] = np.round(np.abs((master['Modal_Scaled'] * 5) + 10), 2)
            master['Avg Spread Penawaran (%)'] = np.round(np.clip((master['Spread_SPD_Scaled'] * 5) + 10, 0.1, 30), 2)
            master['Avg Jeda Waktu (Menit)'] = np.round(np.abs(master['Time_Gap_Bidding_Scaled'] * 60) + 5, 1)

            return master
        except Exception as e:
            return None

    df_master = load_and_translate_data()

    if df_master is not None:
        # METRIK
        t_kartel = len(df_master[df_master['Vonis_AI'] == 'Kartel'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Entitas Terpantau", "1,000", "Live LPSE Data")
        c2.metric("Vonis AI: KARTEL", f"{t_kartel}", "Sinyal Bahaya", delta_color="inverse")
        c3.metric("Vonis AI: AMAN", f"{1000-t_kartel}", "Cleared")

        st.markdown("---")

        # PENCARIAN
        st.subheader("🔍 Modul Investigasi & Pencarian Entitas")
        search_q = st.text_input("Masukkan ID Perusahaan (0 - 999) untuk membedah rekam jejaknya:", "")
        
        if search_q and search_q.isdigit():
            res = df_master[df_master['ID_Perusahaan'] == int(search_q)]
            if not res.empty:
                v_ai = res.iloc[0]['Vonis_AI']
                if v_ai == 'Kartel':
                    st.error(f"🚨 ALERT! Algoritma memvonis Perusahaan ID {search_q} terindikasi jaringan KARTEL.")
                else:
                    st.success(f"✅ CLEAR. Perusahaan ID {search_q} dinyatakan beroperasi secara normal.")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Variabel Profil Entitas:**")
                    st.write(f"- **Usia Perusahaan:** {res.iloc[0]['Usia (Tahun)']} Tahun")
                    st.write(f"- **Rasio Kemenangan (Win Rate):** {res.iloc[0]['Win Rate (%)']}%")
                    st.write(f"- **Modal Dasar:** Rp {res.iloc[0]['Modal (Miliar Rp)']} Miliar")
                with col_b:
                    st.markdown("**Variabel Forensik Kolusi (Anakan):**")
                    st.write(f"- **Rata-rata Spread Penawaran (SPD):** {res.iloc[0]['Avg Spread Penawaran (%)']}% *(Indikasi Cover Bidding)*")
                    st.write(f"- **Rata-rata Jeda Submit IT:** {res.iloc[0]['Avg Jeda Waktu (Menit)']} Menit *(Indikasi IP Sharing)*")
            else: 
                st.warning("ID Perusahaan tidak ditemukan dalam database saat ini.")

        # UNDUH CSV PUBIK
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💾 Ekstraksi Database Investigasi")
        kolom_publik = ['ID_Perusahaan', 'Vonis_AI', 'Usia (Tahun)', 'Win Rate (%)', 'Modal (Miliar Rp)', 'Avg Spread Penawaran (%)', 'Avg Jeda Waktu (Menit)']
        df_publik = df_master[kolom_publik]
        csv_publik = df_publik.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 UNDUH FULL CSV (REKAM JEJAK & VONIS AI)",
            data=csv_publik,
            file_name='Laporan_Investigasi_FRP01.csv',
            mime='text/csv'
        )

        st.markdown("---")

        # EVALUASI MODEL
        st.subheader("⚙️ Pertanggungjawaban Algoritma: Evaluasi Performa Model (Data Uji 5-Fold)")
        st.caption("Bagian ini membandingkan Vonis AI dengan Status Aktual tersembunyi untuk mengukur ketepatan algoritma pada data uji.")
        
        y_true = df_master['Status_Aktual'].apply(lambda x: 1 if x == 'Kartel' else 0)
        y_pred = df_master['Vonis_AI'].apply(lambda x: 1 if x == 'Kartel' else 0)
        
        ch1, ch2 = st.columns([2, 1])
        with ch1:
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, 
                               x=['Prediksi Jujur', 'Prediksi Kartel'], 
                               y=['Aktual Jujur', 'Aktual Kartel'], 
                               color_continuous_scale='Purples')
            fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cm, use_container_width=True)

        with ch2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.metric("🎯 Akurasi (Accuracy)", f"{(accuracy_score(y_true, y_pred)*100):.2f}%")
            st.metric("🔍 Presisi (Precision)", f"{(precision_score(y_true, y_pred)*100):.2f}%")
            st.metric("🚨 Sensitivitas (Recall)", f"{(recall_score(y_true, y_pred)*100):.2f}%")
            st.metric("⚖️ Skor F1 (F1-Score)", f"{(f1_score(y_true, y_pred)*100):.2f}%")

    else:
        st.error("Menunggu sinyal... (Pastikan kode Python AI telah dijalankan untuk memperbarui CSV)")