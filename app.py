import os
import pandas as pd
import plotly.express as px
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL"))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY"))
GEMINI_KEY   = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(
    page_title="Fund Operations Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(ttl=300)
def load(table):
    return pd.DataFrame(supabase.table(table).select("*").execute().data)


funds    = load("funds")
inst     = load("instruments")
cps      = load("counterparties")
ssi      = load("settlement_instructions")
pos      = load("positions")
trades   = load("trades")
breaks   = load("reconciliation_breaks")
nav_hist = load("nav_history")

st.title("Fund Operations Dashboard")
st.caption("Static data  |  Reconciliation  |  Trade monitoring  |  Fund overview  |  AI Assistant")

st.markdown("""
This dashboard simulates the daily monitoring workflow of a middle office team at an institutional
fund servicer (BNP Paribas Securities Services, Citi). These teams manage the operational data
that makes securities settlement possible — ensuring instruments are correctly referenced,
settlement instructions are on file, position records match the custodian, and trades settle on time.

**Static Data** — Instruments, counterparties, and settlement instructions
(SSI — pre-agreed instructions specifying where to deliver or receive securities on settlement date).
A missing or expired SSI means a trade cannot settle and penalties apply under EU regulation (CSDR).  
**Reconciliation Breaks** — Discrepancies between internal position records and custodian records,
tracked from Open through Investigating to Resolved.  
**Trade Monitor** — Last 10 days of trades. Flags overdue settlements and trades with no valid SSI on file.  
**Fund Overview** — AUM, NAV per share, 30-day trend, and asset class breakdown per fund,
with open break exposure as an operational risk indicator.  
**AI Assistant** — Ask any question about the current state of the data in plain English.

*Instrument prices updated daily via GitHub Actions. NAV history refreshed each weekday after European market close.*
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Static Data", "Reconciliation Breaks", "Trade Monitor", "Fund Overview", "AI Assistant"
])


# ── TAB 1 — STATIC DATA ──────────────────────────────────────────────
with tab1:
    st.caption("Reference data that must exist before any trade can settle.")

    st.subheader("Instruments")
    ac_filter = st.multiselect(
        "Asset class", inst["asset_class"].unique(),
        default=list(inst["asset_class"].unique())
    )
    st.dataframe(
        inst[inst["asset_class"].isin(ac_filter)]
            [["isin","name","asset_class","currency","exchange","price","price_date"]],
        use_container_width=True,
    )

    st.divider()

    st.subheader("Counterparties")
    cp_filter = st.multiselect(
        "Type", cps["cp_type"].unique(), default=list(cps["cp_type"].unique())
    )
    def kyc_colour(row):
        if row["kyc_status"] == "Expired": return ["background-color:#C8A0A0"] * len(row)
        if row["kyc_status"] == "Pending": return ["background-color:#C8B888"] * len(row)
        return [""] * len(row)
    st.dataframe(
        cps[cps["cp_type"].isin(cp_filter)]
           [["counterparty_id","name","bic","lei","cp_type","country","kyc_status","kyc_expiry"]]
           .style.apply(kyc_colour, axis=1),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Settlement Instructions (SSI)")
    st.caption("Pre-agreed instructions specifying where to deliver or receive securities for each counterparty-instrument pair.")
    c1, c2 = st.columns(2)
    cp_sel     = c1.selectbox("Counterparty", ["All"] + list(cps["name"].sort_values()))
    ssi_status = c2.multiselect(
        "Status", ["Active","Expired","Pending"],
        default=["Active","Expired","Pending"]
    )
    d = (ssi
         .merge(cps[["counterparty_id","name"]], on="counterparty_id", how="left")
         .merge(inst[["isin","name"]], on="isin", how="left", suffixes=("_cp","_inst")))
    d = d[d["status"].isin(ssi_status)]
    if cp_sel != "All":
        d = d[d["name_cp"] == cp_sel]
    def ssi_colour(row):
        if row["status"] == "Expired": return ["background-color:#C8A0A0"] * len(row)
        if row["status"] == "Pending": return ["background-color:#C8B888"] * len(row)
        return [""] * len(row)
    st.caption("🔴 Expired SSI  🟡 Pending activation  ⬜ Active")
    st.dataframe(
        d[["ssi_id","name_cp","name_inst","instruction_type",
           "custodian_bic","currency","valid_from","valid_to","status"]]
         .style.apply(ssi_colour, axis=1),
        use_container_width=True,
    )
    exp = (ssi["status"] == "Expired").sum()
    pnd = (ssi["status"] == "Pending").sum()
    if exp or pnd:
        st.warning(f"{exp} expired SSI(s) and {pnd} pending SSI(s) require attention.")


# ── TAB 2 — RECONCILIATION BREAKS ────────────────────────────────────
with tab2:
    st.caption("Daily discrepancies between internal position records and custodian records. Each break must be investigated and resolved before the next NAV calculation.")

    d = (breaks
         .merge(funds[["fund_id","name"]], on="fund_id", how="left")
         .merge(inst[["isin","name","asset_class"]], on="isin", how="left",
                suffixes=("_fund","_inst")))

    open_b    = breaks[breaks["status"] == "Open"]
    inv_b     = breaks[breaks["status"] == "Investigating"]
    today_s   = str(pd.Timestamp.today().date())
    res_today = len(breaks[(breaks["status"]=="Resolved")&(breaks["resolved_date"]==today_s)])

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Open Breaks",       len(open_b))
    k2.metric("Investigating",      len(inv_b))
    k3.metric("Total Exposure EUR", f"{open_b['break_value_eur'].sum():,.0f}")
    k4.metric("Resolved Today",     res_today)

    st.divider()
    c1,c2,c3 = st.columns(3)
    fund_sel  = c1.selectbox("Fund", ["All"] + list(funds["name"].sort_values()))
    btype_sel = c2.multiselect(
        "Break type", breaks["break_type"].unique().tolist(),
        default=breaks["break_type"].unique().tolist()
    )
    stat_sel  = c3.multiselect(
        "Status", ["Open","Investigating","Resolved"],
        default=["Open","Investigating"]
    )
    if fund_sel != "All":
        d = d[d["name_fund"] == fund_sel]
    d = d[d["break_type"].isin(btype_sel) & d["status"].isin(stat_sel)]

    def brk_colour(row):
        if row["status"] == "Open":          return ["background-color:#C8A0A0"] * len(row)
        if row["status"] == "Investigating": return ["background-color:#C8B888"] * len(row)
        return ["background-color:#A0B8A0"] * len(row)
    st.caption("🔴 Open  🟡 Investigating  🟢 Resolved")
    st.dataframe(
        d[["break_id","name_fund","isin","name_inst","asset_class",
           "internal_qty","custodian_qty","break_qty","break_value_eur",
           "break_type","status","created_date"]]
         .sort_values("break_value_eur", ascending=False)
         .style.apply(brk_colour, axis=1),
        use_container_width=True,
    )


# ── TAB 3 — TRADE MONITOR ────────────────────────────────────────────
with tab3:
    st.caption("Rolling 10-day trade history. A trade flagged red has missed its settlement date. A trade flagged yellow has no active SSI and will fail on settlement date unless the instruction is set up before the custodian cut-off.")

    d = (trades
         .merge(funds[["fund_id","name"]], on="fund_id", how="left")
         .merge(inst[["isin","name","asset_class"]], on="isin", how="left",
                suffixes=("_fund","_inst"))
         .merge(cps[["counterparty_id","name","bic"]], on="counterparty_id",
                how="left", suffixes=("","_cp")))

    d["settlement_date"] = pd.to_datetime(d["settlement_date"]).dt.date
    today = pd.Timestamp.today().date()
    d["overdue"] = (d["settlement_date"] < today) & (d["status"].isin(["Pending","Failed"]))

    active_keys = set(zip(
        ssi[ssi["status"]=="Active"]["counterparty_id"],
        ssi[ssi["status"]=="Active"]["isin"]
    ))
    d["ssi_missing"] = d.apply(
        lambda r: (r["counterparty_id"], r["isin"]) not in active_keys, axis=1
    )

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Pending", len(d[d["status"]=="Pending"]))
    k2.metric("Settled", len(d[d["status"]=="Settled"]))
    k3.metric("Failed",  len(d[d["status"]=="Failed"]))
    k4.metric("Overdue", int(d["overdue"].sum()))

    st.divider()
    c1,c2 = st.columns(2)
    stat_f   = c1.multiselect(
        "Status", ["Pending","Settled","Failed","Cancelled"],
        default=["Pending","Failed"]
    )
    ssi_only = c2.checkbox("Show only trades with missing Active SSI")
    d = d[d["status"].isin(stat_f)]
    if ssi_only:
        d = d[d["ssi_missing"] == True]

    def trade_colour(row):
        if row["status"]=="Failed" or row["overdue"]: return ["background-color:#C8A0A0"]*len(row)
        if row["ssi_missing"]:                        return ["background-color:#C8B888"]*len(row)
        return [""]*len(row)
    st.caption("🔴 Failed or overdue  🟡 Missing active SSI  ⬜ Normal")
    st.dataframe(
        d[["trade_id","name_fund","name_inst","asset_class","trade_type",
           "quantity","price","trade_date","settlement_date","status",
           "name","bic","ssi_missing","overdue"]]
         .rename(columns={"name":"counterparty","bic":"cp_bic"})
         .sort_values("settlement_date")
         .style.apply(trade_colour, axis=1),
        use_container_width=True,
    )


# ── TAB 4 — FUND OVERVIEW ────────────────────────────────────────────
with tab4:
    st.caption("One card per fund. Open break exposure shows which funds carry the most operational risk today.")

    break_summary = (
        breaks[breaks["status"].isin(["Open","Investigating"])]
        .groupby("fund_id")
        .agg(open_breaks=("break_id","count"), exposure=("break_value_eur","sum"))
        .reset_index()
    )

    for _, f in funds.iterrows():
        fid = f["fund_id"]
        b = break_summary[break_summary["fund_id"] == fid]
        n_breaks = int(b["open_breaks"].values[0]) if len(b) else 0
        exposure  = float(b["exposure"].values[0]) if len(b) else 0.0

        with st.expander(f"{f['name']}  ({f['fund_type']})  —  {f['base_currency']}", expanded=False):
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("AUM (EUR)",    f"{f['aum_eur']/1e6:.1f}M")
            c2.metric("NAV / Share",  f"{f['nav_per_share']:.4f}")
            c3.metric("Fund Type",    f["fund_type"])
            c4.metric("Domicile",     f["domicile"])
            c5.metric("Open Breaks",  n_breaks,
                      delta=None if n_breaks == 0 else f"EUR {exposure:,.0f}",
                      delta_color="inverse")
            c6.metric("NAV Date",     str(f["nav_date"]))

            col_chart, col_nav, col_pos = st.columns([1,1,1])

            fh = nav_hist[nav_hist["fund_id"]==fid].sort_values("nav_date")
            if not fh.empty:
                fig = px.line(fh, x="nav_date", y="nav_per_share", title="NAV per share — 30 day history")
                fig.update_yaxes(range=[fh["nav_per_share"].min()*0.995,
                                        fh["nav_per_share"].max()*1.005])
                col_nav.plotly_chart(fig, use_container_width=True)

            fp = pos[pos["fund_id"]==fid].merge(inst[["isin","asset_class"]], on="isin", how="left")
            if not fp.empty:
                ac = fp.groupby("asset_class")["market_value_eur"].sum().reset_index()
                ac.columns = ["Asset Class","Market Value EUR"]
                col_chart.caption("Portfolio by asset class")
                col_chart.bar_chart(ac.set_index("Asset Class"))
                col_pos.caption("Breakdown (EUR)")
                col_pos.dataframe(ac, use_container_width=True)


# ── TAB 5 — AI ASSISTANT ─────────────────────────────────────────────
with tab5:
    st.caption("Ask any question about the current state of the dashboard data in plain English.")

    def build_context():
        open_breaks = breaks[breaks["status"] == "Open"]
        pending_trades = trades[trades["status"] == "Pending"]
        expired_ssi = ssi[ssi["status"] == "Expired"]
        overdue_trades = trades[
            (pd.to_datetime(trades["settlement_date"]).dt.date < pd.Timestamp.today().date()) &
            (trades["status"].isin(["Pending","Failed"]))
        ]

        ctx = f"""
You are an investment operations analyst assistant. You have access to the following live data
from a fund operations dashboard. Answer questions concisely and accurately based only on this data.

FUNDS ({len(funds)} total):
{funds[["fund_id","name","fund_type","base_currency","aum_eur","nav_per_share"]].to_string(index=False)}

INSTRUMENTS ({len(inst)} total):
{inst[["isin","name","asset_class","currency","price","price_date"]].to_string(index=False)}

COUNTERPARTIES ({len(cps)} total):
{cps[["counterparty_id","name","cp_type","kyc_status"]].to_string(index=False)}

SETTLEMENT INSTRUCTIONS:
- Total: {len(ssi)}
- Active: {(ssi["status"]=="Active").sum()}
- Expired: {(ssi["status"]=="Expired").sum()}
- Pending: {(ssi["status"]=="Pending").sum()}

RECONCILIATION BREAKS:
- Open: {(breaks["status"]=="Open").sum()} — Total exposure EUR: {breaks[breaks["status"]=="Open"]["break_value_eur"].sum():,.0f}
- Investigating: {(breaks["status"]=="Investigating").sum()}
- Resolved: {(breaks["status"]=="Resolved").sum()}
- Break types: {breaks["break_type"].value_counts().to_dict()}

BREAKS BY FUND:
{breaks.groupby("fund_id")["break_id"].count().reset_index().rename(columns={"break_id":"break_count"}).to_string(index=False)}

TRADES (last 10 days):
- Total: {len(trades)}
- Pending: {(trades["status"]=="Pending").sum()}
- Settled: {(trades["status"]=="Settled").sum()}
- Failed: {(trades["status"]=="Failed").sum()}
- Overdue: {len(overdue_trades)}

EXPIRED SSIs:
{expired_ssi[["ssi_id","counterparty_id","isin","valid_to"]].to_string(index=False) if len(expired_ssi) else "None"}
"""
        return ctx

    question = st.text_input("Ask a question about the data:", placeholder="e.g. Which fund has the most open reconciliation breaks?")

    if question:
        with st.spinner("Analysing..."):
            context = build_context()
            prompt = f"{context}\n\nQuestion: {question}\n\nAnswer:"
            try:
                response = model.generate_content(prompt)
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("**Example questions:**")
    examples = [
        "Which fund has the most open reconciliation breaks?",
        "Are there any counterparties with expired KYC?",
        "How many trades are overdue today?",
        "Which SSIs are expired and for which counterparties?",
        "What is the total EUR exposure from open breaks by asset class?",
        "Which fund has the highest AUM?",
    ]
    for ex in examples:
        if st.button(ex):
            with st.spinner("Analysing..."):
                context = build_context()
                prompt = f"{context}\n\nQuestion: {ex}\n\nAnswer:"
                try:
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")