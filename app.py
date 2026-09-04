
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Ecommerce Analytics Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA = Path(__file__).parent / "data"

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    orders = pd.read_csv(DATA / "olist_orders_dataset.csv")
    customers = pd.read_csv(DATA / "olist_customers_dataset.csv")
    items = pd.read_csv(DATA / "olist_order_items_dataset.csv")
    payments = pd.read_csv(DATA / "olist_order_payments_dataset.csv")
    reviews = pd.read_csv(DATA / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(DATA / "olist_products_dataset.csv")
    sellers = pd.read_csv(DATA / "olist_sellers_dataset.csv")

    orders["order_purchase_timestamp"] = pd.to_datetime(
        orders["order_purchase_timestamp"],
        errors="coerce"
    )

    orders["order_delivered_customer_date"] = pd.to_datetime(
        orders["order_delivered_customer_date"],
        errors="coerce"
    )

    orders["order_estimated_delivery_date"] = pd.to_datetime(
        orders["order_estimated_delivery_date"],
        errors="coerce"
    )

    products = products.copy()

    if "product_category_name" in products.columns:
        category_translation = {
            "cama_mesa_banho": "Bed, Bath & Home",
            "esporte_lazer": "Sports & Leisure",
            "moveis_decoracao": "Furniture & Decor",
            "beleza_saude": "Beauty & Health",
            "utilidades_domesticas": "Home & Kitchen",
            "automotivo": "Automotive",
            "informatica_acessorios": "Computers & Accessories",
            "brinquedos": "Toys",
            "relogios_presentes": "Watches & Gifts",
            "telefonia": "Telephony",
            "bebes": "Baby",
            "perfumaria": "Beauty & Fragrance",
            "papelaria": "Stationery",
            "fashion_bolsas_e_acessorios": "Fashion Bags & Accessories",
            "cool_stuff": "Cool Stuff",
            "ferramentas_jardim": "Garden Tools",
            "pet_shop": "Pet Supplies",
            "eletronicos": "Electronics",
            "construcao_ferramentas_construcao": "Construction & Building Tools",
            "eletrodomesticos": "Home Appliances",
            "malas_acessorios": "Luggage & Accessories",
            "consoles_games": "Consoles & Games",
            "moveis_escritorio": "Office Furniture",
            "instrumentos_musicais": "Musical Instruments",
            "eletroportateis": "Small Appliances",
            "casa_construcao": "Home Construction",
            "livros_interesse_geral": "Books — General",
            "fashion_calcados": "Fashion — Shoes",
            "moveis_sala": "Living Room Furniture",
            "climatizacao": "Climate Control",
            "livros_tecnicos": "Technical Books",
            "telefonia_fixa": "Landline Telephones",
            "casa_conforto": "Home Comfort",
            "market_place": "Marketplace",
            "alimentos_bebidas": "Food & Beverages",
            "fashion_roupa_masculina": "Fashion — Men's Clothing",
            "moveis_cozinha_area_de_servico_jantar_e_jardim": "Kitchen, Dining & Garden Furniture",
            "sinalizacao_e_seguranca": "Signage & Security",
            "construcao_ferramentas_seguranca": "Construction & Safety Tools",
            "eletrodomesticos_2": "Home Appliances — Other",
            "construcao_ferramentas_jardim": "Construction & Garden Tools",
            "alimentos": "Food",
            "bebidas": "Beverages",
            "construcao_ferramentas_iluminacao": "Construction & Lighting Tools",
            "agro_industria_e_comercio": "Agriculture & Industry",
            "industria_comercio_e_negocios": "Industry, Commerce & Business",
            "artigos_de_natal": "Christmas Items",
            "audio": "Audio",
            "artes": "Arts",
            "fashion_underwear_e_moda_praia": "Fashion — Underwear & Swimwear",
            "dvds_blu_ray": "DVDs & Blu-ray",
            "moveis_quarto": "Bedroom Furniture",
            "construcao_ferramentas_ferramentas": "Construction Tools",
            "livros_importados": "Imported Books",
            "portateis_casa_forno_e_cafe": "Home, Oven & Coffee Appliances",
            "pcs": "PCs",
            "cine_foto": "Cameras & Photography",
            "fashion_roupa_feminina": "Fashion — Women's Clothing",
            "musica": "Music",
            "artigos_de_festas": "Party Supplies",
            "artes_e_artesanato": "Arts & Crafts",
            "fashion_esporte": "Sports Fashion",
            "flores": "Flowers",
            "fraldas_higiene": "Diapers & Hygiene",
            "la_cuisine": "Kitchen & Dining",
            "moveis_colchao_e_estofado": "Mattresses & Upholstery",
            "portateis_cozinha_e_preparadores_de_alimentos": "Kitchen Appliances & Food Prep",
            "tablets_impressao_imagem": "Tablets, Printing & Imaging",
            "fashion_roupa_infanto_juvenil": "Fashion — Children's Clothing",
            "casa_conforto_2": "Home Comfort — Other",
            "pc_gamer": "Gaming PCs",
            "seguros_e_servicos": "Insurance & Services",
            "cds_dvds_musicais": "Music CDs & DVDs"
        }

        products["category"] = products["product_category_name"].map(
            category_translation
        ).fillna(products["product_category_name"])

    else:
        products["category"] = "unknown"

    return orders, customers, items, payments, reviews, products, sellers


orders, customers, items, payments, reviews, products, sellers = load_data()

# ============================================================
# RFM CUSTOMER SEGMENTATION
# ============================================================

rfm_orders = orders[
    orders["order_status"].isin(["delivered"])
].copy()

rfm_orders = rfm_orders.dropna(
    subset=["order_purchase_timestamp", "customer_id", "order_id"]
)

rfm_items = (
    items.groupby("order_id", as_index=False)
    .agg(order_revenue=("price", "sum"))
)

rfm_orders = rfm_orders.merge(
    rfm_items,
    on="order_id",
    how="left"
)

rfm_orders["order_revenue"] = rfm_orders["order_revenue"].fillna(0)

rfm_reference_date = (
    rfm_orders["order_purchase_timestamp"].max()
    + pd.Timedelta(days=1)
)

rfm = (
    rfm_orders.groupby("customer_id")
    .agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_revenue", "sum")
    )
    .reset_index()
)

rfm["recency"] = (
    rfm_reference_date - rfm["last_purchase"]
).dt.days

rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

rfm["R_Score"] = pd.qcut(
    rfm["recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)

def assign_rfm_segment(row):
    if row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
        return "Champions"
    elif row["R_Score"] >= 3 and row["F_Score"] >= 4:
        return "Loyal Customers"
    elif row["R_Score"] >= 4 and row["F_Score"] <= 2:
        return "New Customers"
    elif row["R_Score"] <= 2 and row["F_Score"] >= 3:
        return "At Risk"
    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "Lost Customers"
    else:
        return "Potential Loyalists"

rfm["segment"] = rfm.apply(assign_rfm_segment, axis=1)

# ============================================================
# ENGLISH PRODUCT CATEGORY LABELS
# ============================================================

CATEGORY_EN = {
    "cama_mesa_banho": "Bed, Bath & Home",
    "esporte_lazer": "Sports & Leisure",
    "moveis_decoracao": "Furniture & Decor",
    "beleza_saude": "Beauty & Health",
    "utilidades_domesticas": "Home & Kitchen",
    "automotivo": "Automotive",
    "informatica_acessorios": "Computers & Accessories",
    "brinquedos": "Toys",
    "relogios_presentes": "Watches & Gifts",
    "telefonia": "Telephony",
    "bebes": "Baby",
    "perfumaria": "Beauty & Fragrance",
    "papelaria": "Stationery",
    "fashion_bolsas_e_acessorios": "Fashion Bags & Accessories",
    "cool_stuff": "Cool Stuff",
    "ferramentas_jardim": "Garden Tools",
    "pet_shop": "Pet Supplies",
    "eletronicos": "Electronics",
    "construcao_ferramentas_construcao": "Construction & Building Tools",
    "eletrodomesticos": "Home Appliances",
    "malas_acessorios": "Luggage & Accessories",
    "consoles_games": "Consoles & Games",
    "moveis_escritorio": "Office Furniture",
    "instrumentos_musicais": "Musical Instruments",
    "eletroportateis": "Small Appliances",
    "casa_construcao": "Home Construction",
    "livros_interesse_geral": "Books — General",
    "fashion_calcados": "Fashion — Shoes",
    "moveis_sala": "Living Room Furniture",
    "climatizacao": "Climate Control",
    "livros_tecnicos": "Technical Books",
    "telefonia_fixa": "Landline Telephones",
    "casa_conforto": "Home Comfort",
    "market_place": "Marketplace",
    "alimentos_bebidas": "Food & Beverages",
    "fashion_roupa_masculina": "Fashion — Men's Clothing",
    "moveis_cozinha_area_de_servico_jantar_e_jardim": "Kitchen, Dining & Garden Furniture",
    "sinalizacao_e_seguranca": "Signage & Security",
    "construcao_ferramentas_seguranca": "Construction & Safety Tools",
    "eletrodomesticos_2": "Home Appliances — Other",
    "construcao_ferramentas_jardim": "Construction & Garden Tools",
    "alimentos": "Food",
    "bebidas": "Beverages",
    "construcao_ferramentas_iluminacao": "Construction & Lighting Tools",
    "agro_industria_e_comercio": "Agriculture & Industry",
    "industria_comercio_e_negocios": "Industry, Commerce & Business",
    "artigos_de_natal": "Christmas Items",
    "audio": "Audio",
    "artes": "Arts",
    "fashion_underwear_e_moda_praia": "Fashion — Underwear & Swimwear",
    "dvds_blu_ray": "DVDs & Blu-ray",
    "moveis_quarto": "Bedroom Furniture",
    "construcao_ferramentas_ferramentas": "Construction Tools",
    "livros_importados": "Imported Books",
    "portateis_casa_forno_e_cafe": "Home, Oven & Coffee Appliances",
    "pcs": "PCs",
    "cine_foto": "Cameras & Photography",
    "fashion_roupa_feminina": "Fashion — Women's Clothing",
    "musica": "Music",
    "artigos_de_festas": "Party Supplies",
    "artes_e_artesanato": "Arts & Crafts",
    "fashion_esporte": "Sports Fashion",
    "flores": "Flowers",
    "fraldas_higiene": "Diapers & Hygiene",
    "la_cuisine": "Kitchen & Dining",
    "moveis_colchao_e_estofado": "Mattresses & Upholstery",
    "portateis_cozinha_e_preparadores_de_alimentos": "Kitchen Appliances & Food Prep",
    "tablets_impressao_imagem": "Tablets, Printing & Imaging",
    "fashion_roupa_infanto_juvenil": "Fashion — Children's Clothing",
    "casa_conforto_2": "Home Comfort — Other",
    "pc_gamer": "Gaming PCs",
    "seguros_e_servicos": "Insurance & Services",
    "cds_dvds_musicais": "Music CDs & DVDs",
}

order_items = items

# ============================================================
# BUILD ANALYTICS DATA
# ============================================================

item_sales = (
    items.groupby("order_id", as_index=False)
    .agg(
        revenue=("price", "sum"),
        items=("order_item_id", "count")
    )
)

sales = orders.merge(item_sales, on="order_id", how="inner")

sales["purchase_month"] = (
    sales["order_purchase_timestamp"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

monthly = (
    sales.groupby("purchase_month", as_index=False)
    .agg(
        orders=("order_id", "nunique"),
        revenue=("revenue", "sum")
    )
)

monthly["average_order_value"] = (
    monthly["revenue"] / monthly["orders"]
)

monthly["revenue_growth_pct"] = (
    monthly["revenue"].pct_change() * 100
)

total_revenue = sales["revenue"].sum()
total_orders = sales["order_id"].nunique()
average_order_value = total_revenue / total_orders

# Customer order counts
customer_orders = (
    orders.groupby("customer_id")["order_id"]
    .nunique()
    .reset_index(name="order_count")
)

customer_segment = customer_orders.copy()

customer_segment["customer_segment"] = np.where(
    customer_segment["order_count"] > 1,
    "Repeat Customer",
    "One-Time Customer"
)

customer_segment_counts = (
    customer_segment["customer_segment"]
    .value_counts()
    .rename_axis("customer_segment")
    .reset_index(name="customers")
)

customer_revenue = (
    sales.merge(
        customer_segment[["customer_id", "customer_segment"]],
        on="customer_id",
        how="left"
    )
    .groupby("customer_segment", as_index=False)
    .agg(revenue=("revenue", "sum"))
)

segment = customer_segment_counts.merge(
    customer_revenue,
    on="customer_segment",
    how="left"
)

segment["customer_pct"] = (
    segment["customers"] / segment["customers"].sum() * 100
)

segment["revenue_pct"] = (
    segment["revenue"] / segment["revenue"].sum() * 100
)

repeat_rows = segment.loc[
    segment["customer_segment"] == "Repeat Customer",
    "customer_pct"
]

repeat_rate = repeat_rows.iloc[0] if not repeat_rows.empty else 0

# Acquisition
first_purchase = (
    orders.groupby("customer_id")["order_purchase_timestamp"]
    .min()
    .reset_index()
)

first_purchase["month"] = (
    first_purchase["order_purchase_timestamp"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

acquisition = (
    first_purchase.groupby("month")
    .size()
    .reset_index(name="new_customers")
)

# Reviews
review_avg = reviews["review_score"].mean()

# Delivery
delivery = orders.dropna(
    subset=[
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
).copy()

delivery["delay_days"] = (
    delivery["order_delivered_customer_date"]
    - delivery["order_estimated_delivery_date"]
).dt.total_seconds() / 86400

delivery["delivery_days"] = (
    delivery["order_delivered_customer_date"]
    - delivery["order_purchase_timestamp"]
).dt.total_seconds() / 86400

delivery["delivery_status"] = np.where(
    delivery["delay_days"] > 0,
    "Late",
    "On Time"
)

late_rate = (
    (delivery["delivery_status"] == "Late").mean() * 100
)

# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(240, 90, 138, 0.08),
            transparent 25%
        ),
        radial-gradient(
            circle at 10% 80%,
            rgba(143, 48, 79, 0.07),
            transparent 28%
        ),
        #0B0A0D;
    color: #F4EFF2;
}

.block-container {
    max-width: 1500px;
    padding: 45px 55px 70px 55px;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

section[data-testid="stSidebar"] {
    background: #100D11;
    border-right: 1px solid #29232A;
}

section[data-testid="stSidebar"] > div {
    padding: 35px 20px;
}

.brand {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 29px;
    color: #F5EEF1;
}

.brand-accent {
    color: #F05A8A;
}

.brand-subtitle {
    color: #756B72;
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-top: 4px;
    margin-bottom: 38px;
}

.nav-label {
    color: #625961;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

section[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 9px 12px;
    margin: 3px 0;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: #171319;
    border-color: #30272F;
}

section[data-testid="stSidebar"] .stRadio label p {
    color: #928891 !important;
    font-size: 13px !important;
}

section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(
        90deg,
        rgba(143,48,79,0.30),
        rgba(240,90,138,0.06)
    );
    border-color: #583040;
    box-shadow: inset 3px 0 0 #F05A8A;
}

section[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #F7EEF2 !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] .stRadio input {
    display: none;
}

.page-eyebrow {
    color: #F05A8A;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.page-title {
    font-family: 'Playfair Display', Georgia, serif;
    color: #F4EFF2;
    font-size: 46px;
    line-height: 1.08;
    font-weight: 500;
    letter-spacing: -1px;
}

.page-description {
    color: #918991;
    font-size: 14px;
    line-height: 1.7;
    max-width: 750px;
    margin-top: 12px;
    margin-bottom: 34px;
}

.kpi-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        145deg,
        rgba(255,255,255,0.035),
        rgba(255,255,255,0.008)
    );
    border: 1px solid #2B252C;
    border-radius: 13px;
    padding: 23px;
    min-height: 125px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.22);
}

.kpi-card::after {
    content: "";
    position: absolute;
    width: 120px;
    height: 120px;
    right: -50px;
    bottom: -65px;
    background: radial-gradient(
        circle,
        rgba(240,90,138,0.15),
        transparent 70%
    );
}

.kpi-label {
    color: #7E747C;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 13px;
}

.kpi-value {
    color: #F6F0F3;
    font-size: 29px;
    font-weight: 600;
}

.kpi-note {
    color: #686068;
    font-size: 10px;
    margin-top: 7px;
}

.section {
    margin-top: 42px;
    margin-bottom: 7px;
}

.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    color: #EEE7EB;
    font-size: 24px;
}

.section-description {
    color: #706870;
    font-size: 12px;
    margin-top: 4px;
}

.insight {
    background: linear-gradient(
        135deg,
        rgba(143,48,79,0.20),
        rgba(240,90,138,0.025)
    );
    border: 1px solid #4A2937;
    border-radius: 13px;
    padding: 22px 25px;
    margin-top: 24px;
}

.insight-heading {
    color: #F05A8A;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.insight-body {
    color: #B7ADB3;
    font-size: 13px;
    line-height: 1.7;
}

/* Remove Plotly/Streamlit white containers */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
[data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove generic white chart/card wrappers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: transparent !important;
    border-color: transparent !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2A252B;
    border-radius: 11px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY THEME
# ============================================================

PLOT_BG = "#0B0A0D"
PAPER_BG = "#0B0A0D"
TEXT = "#B7ADB3"
GRID = "#211D22"
ACCENT = "#F05A8A"
ACCENT_DARK = "#8F304F"

def style_fig(fig, height=390):

    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(
            family="DM Sans",
            color=TEXT,
            size=12
        ),
        margin=dict(l=10, r=10, t=15, b=10),
        hoverlabel=dict(
            bgcolor="#171319",
            font_color="#F4EFF2"
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            color="#756B72"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            color="#756B72"
        )
    )

    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">Ecommerce <span class="brand-accent">Studio</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="nav-label">Explore</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            margin: 8px 0 18px 0;
            padding: 18px 16px;
            border: 1px solid rgba(240,90,138,0.45);
            border-radius: 14px;
            background: linear-gradient(
                135deg,
                rgba(240,90,138,0.16),
                rgba(143,48,79,0.08)
            );
        ">
            <div style="
                color:#F05A8A;
                font-size:10px;
                font-weight:700;
                letter-spacing:1.5px;
                margin-bottom:7px;
            ">NEW • YOUR DATA</div>

            <div style="
                color:#F5EEF1;
                font-size:17px;
                font-weight:700;
                margin-bottom:6px;
            ">🔮 Forecast Your Business</div>

            <div style="
                color:#A89EA4;
                font-size:11px;
                line-height:1.5;
            ">
                Upload your own CSV or Excel data and turn it
                into projections and business insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🔮 Forecast Your Business",
            "Overview",
            "Sales",
            "Customers",
            "Retention",
            "Delivery",
            "Product",
            "Seller",
            "Customer Segmentation",
            "Business Insights"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-meta">'
        'Olist ecommerce dataset<br>'
        'Sales and customer intelligence'
        '</div>',
        unsafe_allow_html=True
    )

# ============================================================
# PAGE HEADER
# ============================================================

titles = {
    "Overview": (
        "Executive Overview",
        "A concise view of sales, customers, retention and delivery performance."
    ),
    "Sales": (
        "Sales Performance",
        "Track revenue, order activity and average order value over time."
    ),
    "Customers": (
        "Customer Analytics",
        "Understand customer growth, segment contribution and acquisition."
    ),
    "Retention": (
        "Customer Retention",
        "Explore the balance between one-time and repeat purchasing behavior."
    ),
    "Delivery": (
        "Delivery Performance",
        "Monitor delivery speed, delays and the relationship with customer reviews."
    ),
    "Product": (
        "Product Performance",
        "Understand product sales, revenue contribution and category performance."
    ),
    "Seller": (
        "Seller Performance",
        "Evaluate seller activity, order volume and revenue contribution."
    ),
    "Customer Segmentation": (
        "Customer Segmentation",
        "Identify your most valuable, loyal, new and at-risk customers using RFM analysis."
    ),
    "Business Insights": (
        "Business Insights",
        "Turn ecommerce performance data into clear business decisions."
    ),
    "Forecast Your Business": (
        "Forecast Your Business",
        "Upload your own ecommerce data and explore projected business performance."
    ),
}

title, description = titles[page]

st.markdown(
    '<div class="page-eyebrow">Ecommerce Analytics Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="page-title">{title}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="page-description">{description}</div>',
    unsafe_allow_html=True
)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Revenue</div>'
            f'<div class="kpi-value">${total_revenue:,.0f}</div>'
            f'<div class="kpi-note">Order item revenue</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Orders</div>'
            f'<div class="kpi-value">{total_orders:,}</div>'
            f'<div class="kpi-note">Completed purchase records</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Average Order</div>'
            f'<div class="kpi-value">${average_order_value:,.0f}</div>'
            f'<div class="kpi-note">Revenue per order</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Repeat Rate</div>'
            f'<div class="kpi-value">{repeat_rate:.1f}%</div>'
            f'<div class="kpi-note">Customers with multiple orders</div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section"><div class="section-title">Revenue trajectory</div>'
        '<div class="section-description">Monthly ecommerce revenue performance.</div></div>',
        unsafe_allow_html=True
    )

    fig = px.area(
        monthly,
        x="purchase_month",
        y="revenue"
    )

    fig.update_traces(
        line=dict(color=ACCENT, width=2),
        fillcolor="rgba(240,90,138,0.10)"
    )

    fig = style_fig(fig, 380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight"><div class="insight-heading">Business highlight</div>'
        f'<div class="insight-body">The strongest month generated '
        f'<b>${monthly["revenue"].max():,.0f}</b> in revenue. '
        f'Repeat customers currently represent <b>{repeat_rate:.1f}%</b> '
        f'of the customer base, highlighting a significant retention opportunity.</div></div>',
        unsafe_allow_html=True
    )

# ============================================================
# SALES
# ============================================================

elif page == "Sales":

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Revenue</div>'
            f'<div class="kpi-value">${total_revenue:,.0f}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Orders</div>'
            f'<div class="kpi-value">{total_orders:,}</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Average Order Value</div>'
            f'<div class="kpi-value">${average_order_value:,.2f}</div></div>',
            unsafe_allow_html=True
        )

    charts = [
        ("Monthly revenue", "revenue"),
        ("Monthly orders", "orders"),
        ("Average order value", "average_order_value")
    ]

    for title_text, column in charts:

        st.markdown(
            f'<div class="section"><div class="section-title">{title_text}</div></div>',
            unsafe_allow_html=True
        )

        fig = px.line(
            monthly,
            x="purchase_month",
            y=column
        )

        fig.update_traces(
            line=dict(color=ACCENT, width=2.5)
        )

        fig = style_fig(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    peak = monthly.loc[monthly["revenue"].idxmax()]

    st.markdown(
        f'<div class="insight"><div class="insight-heading">Peak performance</div>'
        f'<div class="insight-body">The highest-revenue month was '
        f'<b>{peak["purchase_month"].strftime("%B %Y")}</b>, '
        f'with <b>${peak["revenue"]:,.0f}</b> across '
        f'<b>{peak["orders"]:,}</b> orders.</div></div>',
        unsafe_allow_html=True
    )

# ============================================================
# CUSTOMERS
# ============================================================

elif page == "Customers":

    total_customers = customer_segment_counts["customers"].sum()

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Customers</div>'
            f'<div class="kpi-value">{total_customers:,}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Repeat Rate</div>'
            f'<div class="kpi-value">{repeat_rate:.1f}%</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Average Review</div>'
            f'<div class="kpi-value">{review_avg:.2f} / 5</div></div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section"><div class="section-title">Customer acquisition</div>'
        '<div class="section-description">New customers by first purchase month.</div></div>',
        unsafe_allow_html=True
    )

    fig = px.area(
        acquisition,
        x="month",
        y="new_customers"
    )

    fig.update_traces(
        line=dict(color=ACCENT, width=2),
        fillcolor="rgba(240,90,138,0.10)"
    )

    fig = style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section"><div class="section-title">Customer segment summary</div>'
        '<div class="section-description">Customer distribution and revenue contribution.</div></div>',
        unsafe_allow_html=True
    )

    display_segment = segment.copy()

    display_segment["revenue"] = display_segment["revenue"].map(
        lambda x: f"${x:,.0f}"
    )

    display_segment["customer_pct"] = display_segment["customer_pct"].map(
        lambda x: f"{x:.1f}%"
    )

    display_segment["revenue_pct"] = display_segment["revenue_pct"].map(
        lambda x: f"{x:.1f}%"
    )

    display_segment = display_segment[
        [
            "customer_segment",
            "customers",
            "customer_pct",
            "revenue",
            "revenue_pct"
        ]
    ]

    # Use a styled HTML table instead of Streamlit's white dataframe.
    rows = ""

    for _, row in display_segment.iterrows():
        rows += (
            f"<tr>"
            f"<td>{row['customer_segment']}</td>"
            f"<td>{row['customers']:,}</td>"
            f"<td>{row['customer_pct']}</td>"
            f"<td>{row['revenue']}</td>"
            f"<td>{row['revenue_pct']}</td>"
            f"</tr>"
        )

    table_html = f"""
    <div style="
        margin-top:14px;
        border:1px solid #2A252B;
        border-radius:12px;
        overflow:hidden;
        background:#111014;
    ">
        <table style="
            width:100%;
            border-collapse:collapse;
            color:#C8BEC5;
            font-family:DM Sans,sans-serif;
            font-size:13px;
        ">
            <thead>
                <tr style="
                    background:#171319;
                    color:#81767F;
                    text-transform:uppercase;
                    letter-spacing:1.3px;
                    font-size:9px;
                ">
                    <th style="padding:15px;text-align:left;">Segment</th>
                    <th style="padding:15px;text-align:right;">Customers</th>
                    <th style="padding:15px;text-align:right;">Customer Share</th>
                    <th style="padding:15px;text-align:right;">Revenue</th>
                    <th style="padding:15px;text-align:right;">Revenue Share</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """

    st.markdown(table_html, unsafe_allow_html=True)

# ============================================================
# RETENTION
# ============================================================

elif page == "Retention":

    st.markdown(
        '<div class="section"><div class="section-title">Customer retention</div>'
        '<div class="section-description">One-time versus repeat purchasing behavior.</div></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            segment,
            names="customer_segment",
            values="customers",
            hole=0.62
        )

        fig.update_traces(
            marker=dict(
                colors=[ACCENT_DARK, ACCENT]
            ),
            textfont=dict(color="#F4EFF2")
        )

        fig = style_fig(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            segment,
            x="customer_segment",
            y="revenue"
        )

        fig.update_traces(
            marker_color=ACCENT
        )

        fig = style_fig(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight"><div class="insight-heading">Retention opportunity</div>'
        f'<div class="insight-body">Only <b>{repeat_rate:.1f}%</b> of customers '
        f'are classified as repeat customers. Increasing repeat purchasing '
        f'could materially improve customer lifetime value.</div></div>',
        unsafe_allow_html=True
    )



# ============================================================
# CUSTOMER SEGMENTATION
# ============================================================

elif page == "Customer Segmentation":

    st.markdown(
        '<div class="eyebrow">CUSTOMER INTELLIGENCE</div>'
        '<div class="page-title">Customer Segmentation</div>'
        '<div class="page-description">'
        'Understand customer value, loyalty and retention opportunities using RFM analysis.'
        '</div>',
        unsafe_allow_html=True
    )

    segment_summary = (
        rfm.groupby("segment", as_index=False)
        .agg(
            customers=("customer_id", "count"),
            revenue=("monetary", "sum"),
            average_order_value=("monetary", "mean")
        )
        .sort_values("customers", ascending=False)
    )

    champions = int(
        segment_summary.loc[
            segment_summary["segment"] == "Champions",
            "customers"
        ].sum()
    )

    at_risk = int(
        segment_summary.loc[
            segment_summary["segment"] == "At Risk",
            "customers"
        ].sum()
    )

    lost = int(
        segment_summary.loc[
            segment_summary["segment"] == "Lost Customers",
            "customers"
        ].sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Customers Analyzed</div>'
            f'<div class="kpi-value">{len(rfm):,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Champions</div>'
            f'<div class="kpi-value">{champions:,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">At Risk</div>'
            f'<div class="kpi-value">{at_risk:,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Lost Customers</div>'
            f'<div class="kpi-value">{lost:,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Customer segment distribution</div>'
        '<div class="section-description">'
        'The size of each RFM customer segment.'
        '</div></div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        segment_summary.sort_values("customers"),
        x="customers",
        y="segment",
        orientation="h"
    )

    fig.update_traces(marker_color=ACCENT)

    fig = style_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Revenue by customer segment</div>'
        '<div class="section-description">'
        'Total item revenue associated with each RFM segment.'
        '</div></div>',
        unsafe_allow_html=True
    )

    revenue_summary = segment_summary.sort_values("revenue")

    fig = px.bar(
        revenue_summary,
        x="revenue",
        y="segment",
        orientation="h"
    )

    fig.update_traces(marker_color=ACCENT)

    fig = style_fig(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    total_customers_rfm = len(rfm)

    at_risk_pct = at_risk / total_customers_rfm * 100
    champions_pct = champions / total_customers_rfm * 100

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Customer intelligence</div>'
        f'<div class="insight-body">'
        f'<b>{at_risk_pct:.1f}%</b> of analyzed customers are currently classified '
        f'as <b>At Risk</b>, while <b>{champions_pct:.1f}%</b> are '
        f'<b>Champions</b>. The strongest opportunity is to protect high-value '
        f'customers while creating campaigns that move newer customers toward '
        f'repeat purchasing.'
        f'</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# PRODUCT PERFORMANCE
# ============================================================

elif page == "Product":

    st.markdown(
        '<div class="eyebrow">PRODUCT INTELLIGENCE</div>'
        '<div class="page-title">Product Performance</div>'
        '<div class="page-description">'
        'Understand which product categories drive revenue and demand.'
        '</div>',
        unsafe_allow_html=True
    )

    product_analysis = order_items.merge(
        products[
            ["product_id", "category"]
        ],
        on="product_id",
        how="left"
    )

    product_analysis["category"] = (
        product_analysis["category"]
        .fillna("Unknown")
    )

    product_summary = (
        product_analysis
        .groupby("category")
        .agg(
            revenue=("price", "sum"),
            orders=("order_id", "nunique"),
            items=("order_item_id", "count")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Categories</div>'
            f'<div class="kpi-value">{len(product_summary):,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Product Revenue</div>'
            f'<div class="kpi-value">${product_summary["revenue"].sum():,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Items Sold</div>'
            f'<div class="kpi-value">{len(product_analysis):,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Top product categories</div>'
        '<div class="section-description">'
        'Revenue contribution from the highest-performing categories.'
        '</div></div>',
        unsafe_allow_html=True
    )

    top_products = (
        product_summary
        .head(12)
        .sort_values("revenue")
    )

    fig = px.bar(
        top_products,
        x="revenue",
        y="category",
        orientation="h"
    )

    fig.update_traces(marker_color=ACCENT)

    fig = style_fig(fig, 420)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Product insight</div>'
        f'<div class="insight-body">'
        f'The leading product category generated '
        f'<b>${product_summary.iloc[0]["revenue"]:,.0f}</b> in item revenue.'
        f'</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# SELLER PERFORMANCE
# ============================================================

elif page == "Seller":

    st.markdown(
        '<div class="eyebrow">SELLER INTELLIGENCE</div>'
        '<div class="page-title">Seller Performance</div>'
        '<div class="page-description">'
        'Identify the sellers contributing the most revenue and order volume.'
        '</div>',
        unsafe_allow_html=True
    )

    seller_summary = (
        order_items
        .groupby("seller_id")
        .agg(
            revenue=("price", "sum"),
            orders=("order_id", "nunique"),
            items=("order_item_id", "count")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Active Sellers</div>'
            f'<div class="kpi-value">{len(seller_summary):,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Seller Revenue</div>'
            f'<div class="kpi-value">${seller_summary["revenue"].sum():,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Orders</div>'
            f'<div class="kpi-value">{seller_summary["orders"].sum():,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Top sellers</div>'
        '<div class="section-description">'
        'Highest-performing sellers ranked by item revenue.'
        '</div></div>',
        unsafe_allow_html=True
    )

    top_sellers = (
        seller_summary
        .head(15)
        .sort_values("revenue")
        .copy()
    )

    top_sellers["seller_label"] = [
        f"Seller {i}"
        for i in range(1, len(top_sellers) + 1)
    ]

    fig = px.bar(
        top_sellers,
        x="revenue",
        y="seller_label",
        orientation="h",
        hover_data={"seller_label": False, "seller_id": True}
    )

    fig.update_traces(marker_color=ACCENT)

    fig = style_fig(fig, 450)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Seller insight</div>'
        f'<div class="insight-body">'
        f'The top seller generated '
        f'<b>${seller_summary.iloc[0]["revenue"]:,.0f}</b> '
        f'in item revenue.'
        f'</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# DELIVERY
# ============================================================

elif page == "Delivery":

    avg_delivery = delivery["delivery_days"].mean()
    avg_delay = delivery["delay_days"].mean()
    late_orders = (delivery["delivery_status"] == "Late").sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Delivery Orders</div>'
            f'<div class="kpi-value">{len(delivery):,}</div></div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Late Rate</div>'
            f'<div class="kpi-value">{late_rate:.1f}%</div></div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Avg Delivery</div>'
            f'<div class="kpi-value">{avg_delivery:.1f} days</div></div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Avg Delay</div>'
            f'<div class="kpi-value">{avg_delay:.1f} days</div></div>',
            unsafe_allow_html=True
        )

    delivery_summary = (
        delivery.groupby("delivery_status", as_index=False)
        .agg(
            orders=("order_id", "count"),
            average_delivery_days=("delivery_days", "mean")
        )
    )

    st.markdown(
        '<div class="section"><div class="section-title">Delivery status</div>'
        '<div class="section-description">Orders delivered on time versus late.</div></div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        delivery_summary,
        x="delivery_status",
        y="orders"
    )

    fig.update_traces(marker_color=ACCENT)

    fig = style_fig(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section"><div class="section-title">Delivery time distribution</div></div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(
        delivery,
        x="delivery_days",
        nbins=40
    )

    fig.update_traces(
        marker_color=ACCENT
    )

    fig = style_fig(fig, 350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight"><div class="insight-heading">Delivery insight</div>'
        f'<div class="insight-body"><b>{late_orders:,}</b> orders were delivered '
        f'after the estimated delivery date, representing <b>{late_rate:.1f}%</b> '
        f'of orders with delivery information.</div></div>',
        unsafe_allow_html=True
    )


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.markdown(
        '<div class="eyebrow">BUSINESS INTELLIGENCE</div>'
        '<div class="page-title">Business Insights</div>'
        '<div class="page-description">'
        'Turn ecommerce performance data into clear business decisions.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CORE BUSINESS METRICS
    # --------------------------------------------------------

    total_revenue_bi = order_items["price"].sum()

    repeat_rate_bi = repeat_rate

    # --------------------------------------------------------
    # DELIVERY METRICS
    # Calculate independently so this page does not depend
    # on variables created inside another page.
    # --------------------------------------------------------

    delivery_orders_bi = orders[
        orders["order_status"] == "delivered"
    ].copy()

    delivery_orders_bi = delivery_orders_bi.dropna(
        subset=[
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )

    delivery_orders_bi["late"] = (
        delivery_orders_bi["order_delivered_customer_date"]
        > delivery_orders_bi["order_estimated_delivery_date"]
    )

    late_orders_bi = int(
        delivery_orders_bi["late"].sum()
    )

    delivery_count_bi = len(delivery_orders_bi)

    late_rate_bi = (
        late_orders_bi / delivery_count_bi * 100
        if delivery_count_bi > 0 else 0
    )

    # --------------------------------------------------------
    # TOP PRODUCT CATEGORY
    # --------------------------------------------------------

    business_product = (
        order_items
        .merge(
            products[["product_id", "category"]],
            on="product_id",
            how="left"
        )
        .groupby("category", as_index=False)
        .agg(
            revenue=("price", "sum"),
            items=("order_item_id", "count")
        )
        .sort_values("revenue", ascending=False)
    )

    top_category = business_product.iloc[0]["category"]
    top_category_revenue = business_product.iloc[0]["revenue"]

    # --------------------------------------------------------
    # SELLER PERFORMANCE
    # --------------------------------------------------------

    business_seller = (
        order_items
        .groupby("seller_id", as_index=False)
        .agg(
            revenue=("price", "sum")
        )
        .sort_values("revenue", ascending=False)
    )

    top_seller_revenue = business_seller.iloc[0]["revenue"]

    # --------------------------------------------------------
    # RFM CUSTOMER SEGMENTS
    # --------------------------------------------------------

    at_risk_count_bi = int(
        (rfm["segment"] == "At Risk").sum()
    )

    champions_count_bi = int(
        (rfm["segment"] == "Champions").sum()
    )

    rfm_total_bi = len(rfm)

    at_risk_pct_bi = (
        at_risk_count_bi / rfm_total_bi * 100
        if rfm_total_bi > 0 else 0
    )

    champions_pct_bi = (
        champions_count_bi / rfm_total_bi * 100
        if rfm_total_bi > 0 else 0
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Revenue</div>'
            f'<div class="kpi-value">${total_revenue_bi:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Repeat Rate</div>'
            f'<div class="kpi-value">{repeat_rate_bi:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">At Risk Customers</div>'
            f'<div class="kpi-value">{at_risk_count_bi:,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Late Delivery Rate</div>'
            f'<div class="kpi-value">{late_rate_bi:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Executive summary</div>'
        '<div class="section-description">'
        'The most important signals from the ecommerce dataset.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Revenue driver</div>'
        f'<div class="insight-body">'
        f'<b>{top_category}</b> is the highest-revenue product category, '
        f'generating <b>${top_category_revenue:,.0f}</b> in item revenue.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Customer opportunity</div>'
        f'<div class="insight-body">'
        f'<b>{at_risk_pct_bi:.1f}%</b> of analyzed customers are classified '
        f'as <b>At Risk</b>, while <b>{champions_pct_bi:.1f}%</b> are '
        f'<b>Champions</b>. Retention efforts should prioritize customers '
        f'showing signs of declining engagement.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Operational opportunity</div>'
        f'<div class="insight-body">'
        f'<b>{late_rate_bi:.1f}%</b> of orders with delivery information '
        f'were delivered after the estimated date. Improving delivery '
        f'reliability could strengthen the customer experience.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Seller benchmark</div>'
        f'<div class="insight-body">'
        f'The highest-performing seller generated '
        f'<b>${top_seller_revenue:,.0f}</b> in item revenue. '
        f'This provides a useful benchmark for comparing seller activity.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # RECOMMENDED ACTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Recommended actions</div>'
        '<div class="section-description">'
        'Business actions suggested by the observed performance signals.'
        '</div></div>',
        unsafe_allow_html=True
    )

    actions = [
        (
            "01",
            "Protect at-risk customers",
            f"Prioritize the {at_risk_count_bi:,} customers currently classified as At Risk with targeted retention campaigns."
        ),
        (
            "02",
            "Increase repeat purchasing",
            f"With a repeat customer rate of {repeat_rate_bi:.1f}%, focus on converting one-time buyers into repeat customers."
        ),
        (
            "03",
            "Protect top categories",
            f"Prioritize inventory and merchandising attention for {top_category}, the leading revenue category."
        ),
        (
            "04",
            "Improve delivery reliability",
            f"Investigate the {late_rate_bi:.1f}% late-delivery rate to reduce operational friction and improve customer satisfaction."
        )
    ]

    for number, heading, body in actions:

        st.markdown(
            f'<div class="insight">'
            f'<div class="insight-heading">{number} · {heading}</div>'
            f'<div class="insight-body">{body}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div style="margin-top:60px;padding-top:18px;border-top:1px solid #29242A;'
    'color:#555057;font-size:9px;letter-spacing:1px;">'
    'ECOMMERCE ANALYTICS STUDIO · OLIST DATASET'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

if page == "Business Insights":

    st.markdown(
        '<div class="eyebrow">BUSINESS INTELLIGENCE</div>'
        '<div class="page-title">Business Insights</div>'
        '<div class="page-description">'
        'Turn ecommerce performance data into clear business decisions.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PERFORMANCE SNAPSHOT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Performance snapshot</div>'
        '<div class="section-description">'
        'A high-level view of the most important business metrics.'
        '</div></div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Total Revenue</div>'
            f'<div class="kpi-value">${total_revenue:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Orders</div>'
            f'<div class="kpi-value">{total_orders:,}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Average Order</div>'
            f'<div class="kpi-value">${average_order_value:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Repeat Rate</div>'
            f'<div class="kpi-value">{repeat_rate:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # CUSTOMER OPPORTUNITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Customer opportunity</div>'
        '<div class="section-description">'
        'Customer behavior highlights where retention efforts may have the greatest impact.'
        '</div></div>',
        unsafe_allow_html=True
    )

    at_risk_count = int(
        rfm["segment"].eq("At Risk").sum()
    )

    champions_count = int(
        rfm["segment"].eq("Champions").sum()
    )

    at_risk_pct = (
        at_risk_count / len(rfm) * 100
        if len(rfm) > 0 else 0
    )

    champions_pct = (
        champions_count / len(rfm) * 100
        if len(rfm) > 0 else 0
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Retention opportunity</div>'
        f'<div class="insight-body">'
        f'<b>{at_risk_pct:.1f}%</b> of analyzed customers are currently '
        f'classified as <b>At Risk</b>, while <b>{champions_pct:.1f}%</b> '
        f'are classified as <b>Champions</b>. '
        f'This suggests an opportunity to protect high-value customers '
        f'while encouraging more customers toward repeat purchasing.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PRODUCT OPPORTUNITY
    # --------------------------------------------------------

    if "product_summary" in locals() and len(product_summary) > 0:

        top_category = product_summary.iloc[0]["category"]
        top_category_revenue = product_summary.iloc[0]["revenue"]

        st.markdown(
            f'<div class="insight">'
            f'<div class="insight-heading">Product opportunity</div>'
            f'<div class="insight-body">'
            f'<b>{top_category}</b> is the leading product category, '
            f'generating <b>${top_category_revenue:,.0f}</b> in item revenue. '
            f'This category represents a strong area to monitor for inventory, '
            f'promotions and future growth.'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # SELLER OPPORTUNITY
    # --------------------------------------------------------

    if "seller_summary" in locals() and len(seller_summary) > 0:

        top_seller_revenue = seller_summary.iloc[0]["revenue"]

        st.markdown(
            f'<div class="insight">'
            f'<div class="insight-heading">Seller performance</div>'
            f'<div class="insight-body">'
            f'The highest-performing seller generated '
            f'<b>${top_seller_revenue:,.0f}</b> in item revenue. '
            f'High-performing sellers can provide useful benchmarks '
            f'for understanding marketplace performance.'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # DELIVERY OPPORTUNITY
    # --------------------------------------------------------

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">Operational opportunity</div>'
        f'<div class="insight-body">'
        f'<b>{late_rate:.1f}%</b> of orders with delivery information '
        f'were delivered after the estimated delivery date. '
        f'Reducing delivery delays could improve the overall customer experience.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # EXECUTIVE TAKEAWAY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">Executive takeaway</div>'
        '<div class="section-description">'
        'The most important themes emerging from the ecommerce data.'
        '</div></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="insight">'
        f'<div class="insight-heading">What the data suggests</div>'
        f'<div class="insight-body">'
        f'The business has established revenue and order activity, but the '
        f'largest opportunities appear to be around <b>customer retention</b>, '
        f'<b>protecting valuable customers</b>, and <b>improving operational performance</b>. '
        f'Product and seller performance can then be used to identify where '
        f'growth and marketplace resources should be concentrated.'
        f'</div></div>',
        unsafe_allow_html=True
    )



# ============================================================
# FORECAST YOUR BUSINESS
# ============================================================

if page == "🔮 Forecast Your Business":

    st.markdown(
        '<div class="eyebrow">YOUR DATA • YOUR BUSINESS</div>'
        '<div class="page-title">Forecast Your Business</div>'
        '<div class="page-description">'
        'Upload your own data and turn historical performance into '
        'simple forward-looking projections.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            padding:20px;
            margin:20px 0 25px 0;
            border-radius:16px;
            border:1px solid #302830;
            background:#110E12;
        ">
            <div style="
                font-size:18px;
                font-weight:700;
                color:#F5EEF1;
                margin-bottom:8px;
            ">Bring your own business data</div>

            <div style="
                color:#A89EA4;
                font-size:13px;
                line-height:1.6;
            ">
                Upload a CSV or Excel file. We'll help you identify
                the important columns, visualize the historical trend,
                and create a simple projection.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="For best results, include a date column and a numeric business metric such as sales, revenue, orders, or customers."
    )

    if uploaded_file is None:

        st.info(
            "👆 Upload a CSV or Excel file above to start exploring your business data."
        )

    else:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_excel(uploaded_file)

            if user_df.empty:
                st.error("The uploaded file is empty.")
                st.stop()

            st.success(
                f"✅ Loaded {len(user_df):,} rows and {len(user_df.columns):,} columns."
            )

            st.markdown("### 1. Preview your data")

            st.dataframe(
                user_df.head(10),
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Detect possible date columns
            # ------------------------------------------------

            date_candidates = []

            for col in user_df.columns:

                col_lower = str(col).lower()

                if any(
                    keyword in col_lower
                    for keyword in [
                        "date",
                        "time",
                        "month",
                        "day",
                        "year"
                    ]
                ):
                    date_candidates.append(col)

            # Also test object columns for date-like values
            for col in user_df.columns:

                if col not in date_candidates:

                    if user_df[col].dtype == "object":

                        sample = user_df[col].dropna().head(20)

                        if len(sample) > 0:

                            parsed = pd.to_datetime(
                                sample,
                                errors="coerce"
                            )

                            if parsed.notna().mean() >= 0.7:
                                date_candidates.append(col)

            date_candidates = list(dict.fromkeys(date_candidates))

            # ------------------------------------------------
            # Numeric columns
            # ------------------------------------------------

            numeric_columns = list(
                user_df.select_dtypes(
                    include=np.number
                ).columns
            )

            if not numeric_columns:

                st.error(
                    "I couldn't find a numeric column to analyze. "
                    "Your file needs at least one numeric metric such as sales, revenue, orders, or customers."
                )
                st.stop()

            st.markdown("### 2. Choose what you want to forecast")

            col1, col2 = st.columns(2)

            with col1:

                if date_candidates:

                    date_col = st.selectbox(
                        "Date column",
                        date_candidates
                    )

                else:

                    date_col = None

                    st.warning(
                        "No obvious date column was detected. "
                        "We'll use the row order as the timeline."
                    )

            with col2:

                metric_col = st.selectbox(
                    "Metric to forecast",
                    numeric_columns
                )

            # ------------------------------------------------
            # Prepare data
            # ------------------------------------------------

            forecast_df = user_df.copy()

            if date_col is not None:

                forecast_df[date_col] = pd.to_datetime(
                    forecast_df[date_col],
                    errors="coerce"
                )

                forecast_df[metric_col] = pd.to_numeric(
                    forecast_df[metric_col],
                    errors="coerce"
                )

                forecast_df = forecast_df.dropna(
                    subset=[date_col, metric_col]
                )

                if len(forecast_df) < 5:

                    st.error(
                        "There aren't enough valid date/metric observations "
                        "to create a projection."
                    )
                    st.stop()

                # Monthly aggregation keeps noisy transaction-level
                # datasets easier to understand.
                forecast_df["period"] = (
                    forecast_df[date_col]
                    .dt.to_period("M")
                    .dt.to_timestamp()
                )

                trend_df = (
                    forecast_df
                    .groupby("period", as_index=False)[metric_col]
                    .sum()
                    .sort_values("period")
                )

                period_col = "period"

            else:

                forecast_df[metric_col] = pd.to_numeric(
                    forecast_df[metric_col],
                    errors="coerce"
                )

                forecast_df = forecast_df.dropna(
                    subset=[metric_col]
                )

                if len(forecast_df) < 5:

                    st.error(
                        "There aren't enough numeric observations "
                        "to create a projection."
                    )
                    st.stop()

                trend_df = forecast_df[
                    [metric_col]
                ].reset_index(drop=True)

                trend_df["period"] = np.arange(
                    1,
                    len(trend_df) + 1
                )

                period_col = "period"

            # ------------------------------------------------
            # Historical trend
            # ------------------------------------------------

            st.markdown("### 3. Historical performance")

            chart_data = trend_df.copy()

            st.line_chart(
                chart_data.set_index(period_col)[metric_col],
                use_container_width=True
            )

            # ------------------------------------------------
            # Forecast
            # ------------------------------------------------

            st.markdown("### 4. Projection")

            y = trend_df[metric_col].astype(float).values
            x = np.arange(len(y), dtype=float)

            slope, intercept = np.polyfit(x, y, 1)

            history_count = len(y)

            # Six future periods
            future_x = np.arange(
                history_count,
                history_count + 6,
                dtype=float
            )

            forecast_values = (
                intercept + slope * future_x
            )

            forecast_values = np.maximum(
                forecast_values,
                0
            )

            if date_col is not None:

                last_period = trend_df[period_col].max()

                future_periods = pd.date_range(
                    last_period + pd.offsets.MonthBegin(1),
                    periods=6,
                    freq="MS"
                )

            else:

                future_periods = [
                    f"Future {i}"
                    for i in range(1, 7)
                ]

            forecast_result = pd.DataFrame(
                {
                    "period": future_periods,
                    "Projected": forecast_values
                }
            )

            st.line_chart(
                forecast_result.set_index("period")["Projected"],
                use_container_width=True
            )

            # ------------------------------------------------
            # KPI cards
            # ------------------------------------------------

            latest_value = float(y[-1])
            projected_value = float(forecast_values[-1])

            if latest_value != 0:

                projected_change = (
                    (projected_value - latest_value)
                    / abs(latest_value)
                    * 100
                )

            else:

                projected_change = 0

            k1, k2, k3 = st.columns(3)

            with k1:

                st.metric(
                    "Latest value",
                    f"{latest_value:,.2f}"
                )

            with k2:

                st.metric(
                    "Projected value",
                    f"{projected_value:,.2f}"
                )

            with k3:

                st.metric(
                    "Projected change",
                    f"{projected_change:+.1f}%"
                )

            # ------------------------------------------------
            # Business insights
            # ------------------------------------------------

            st.markdown("### 5. Business insights")

            if slope > 0:

                direction = "upward"

                insight_text = (
                    f"Your selected metric shows an **upward trend** "
                    f"over the available history. The projection continues "
                    f"that direction into the next six periods."
                )

            elif slope < 0:

                direction = "downward"

                insight_text = (
                    f"Your selected metric shows a **downward trend** "
                    f"over the available history. The projection continues "
                    f"that direction into the next six periods."
                )

            else:

                direction = "stable"

                insight_text = (
                    f"Your selected metric is relatively **stable** "
                    f"over the available history, so the projection "
                    f"remains close to its recent level."
                )

            st.markdown(
                f"""
                <div style="
                    padding:20px;
                    border-radius:14px;
                    border:1px solid #302830;
                    background:#110E12;
                    line-height:1.7;
                ">
                    <div style="
                        color:#F05A8A;
                        font-weight:700;
                        font-size:11px;
                        letter-spacing:1px;
                        margin-bottom:8px;
                    ">AUTOMATED INSIGHT</div>

                    <div style="
                        color:#E8E0E4;
                        font-size:14px;
                    ">
                        {insight_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "Projection method: linear trend based on the uploaded historical data. "
                "This is an analytical estimate, not a guarantee of future performance."
            )

        except Exception as e:

            st.error(
                "I couldn't analyze this file. "
                "Please check that the file contains readable columns and valid data."
            )

            st.exception(e)
