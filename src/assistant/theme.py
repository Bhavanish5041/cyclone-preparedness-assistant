"""Warm Chocolate Editorial Design System, Color Tokens, and Component Builders.

Provides:
- Warm chocolate color palette:
  - Main background: #211914
  - Secondary background: #2B211B
  - Card background: #342820
  - Elevated surfaces: #403127
  - Primary text: #F3E9DC
  - Secondary text: #BDAE9C
  - Muted text: #918170
  - Primary accent: #C18B62
  - Secondary accent: #D6B28C
  - Borders: #514033
- Editorial typography:
  - Headings: DM Serif Display
  - Interface and body: DM Sans
  - Tabular numerals: DM Sans with tabular-nums
- Strictly zero emojis, zero em dashes, zero glowing borders.
"""

from typing import Any, Dict, Optional

# Warm Chocolate Palette Tokens
THEME_COLORS = {
    "background_base": "#211914",
    "background_secondary": "#2B211B",
    "surface_card": "#342820",
    "surface_elevated": "#403127",
    "text_primary": "#F3E9DC",
    "text_secondary": "#BDAE9C",
    "text_muted": "#918170",
    "accent_primary": "#C18B62",
    "accent_secondary": "#D6B28C",
    "border_subtle": "#514033",
    "border_highlight": "#C18B62",
    "status_critical": "#C95D4A",
    "status_specialized": "#D48B47",
    "status_moderate": "#A88E75",
    "status_adequate": "#7A8B63",
}

# Coordinated palette for vulnerability archetypes
ARCHETYPE_COLORS: Dict[int, Dict[str, str]] = {
    0: {"name": "Urban and Peri-Urban Corridor", "hex": "#7A8B63", "badge": "success"},
    1: {"name": "Remote Foothill Communities", "hex": "#D48B47", "badge": "warning"},
    2: {"name": "Vulnerable Rural Interior", "hex": "#A76F4E", "badge": "secondary"},
    3: {"name": "High-Exposure Coastal Belt", "hex": "#C95D4A", "badge": "danger"},
}

# Authoritative palette for decision support priority bands
BAND_COLORS: Dict[str, Dict[str, str]] = {
    "BAND_1_CRITICAL": {
        "name": "Band 1: Critical Priority",
        "short_label": "Band 1",
        "hex": "#C95D4A",
        "badge_class": "badge-band-1",
        "description": "Immediate intervention; low preparedness baseline (<40) or compound vulnerability.",
    },
    "BAND_2_SPECIALIZED": {
        "name": "Band 2: Specialized Readiness",
        "short_label": "Band 2",
        "hex": "#D48B47",
        "badge_class": "badge-band-2",
        "description": "Medium preparedness (40-60) with targeted coastal proximity (<5km), shelter deficit, or medical deficit.",
    },
    "BAND_3_MODERATE": {
        "name": "Band 3: Moderate Priority",
        "short_label": "Band 3",
        "hex": "#A88E75",
        "badge_class": "badge-band-3",
        "description": "Secondary readiness; standard medium preparedness baseline.",
    },
    "BAND_4_ADEQUATE": {
        "name": "Band 4: Adequate Baseline",
        "short_label": "Band 4",
        "hex": "#7A8B63",
        "badge_class": "badge-band-4",
        "description": "Adequate preparedness (>60); standard vigilance and maintenance.",
    },
}

# AHP Preparedness Tier metadata and colors
TIER_COLORS: Dict[str, Dict[str, str]] = {
    "Low": {
        "name": "Low Preparedness (<40)",
        "hex": "#C95D4A",
        "badge_class": "badge-tier-low",
        "description": "AHP Score < 40.0; critical infrastructure or resource deficits.",
    },
    "Medium": {
        "name": "Medium Preparedness (40-60)",
        "hex": "#D48B47",
        "badge_class": "badge-tier-medium",
        "description": "AHP Score 40.0-60.0; moderate baseline readiness.",
    },
    "High": {
        "name": "High Preparedness (>60)",
        "hex": "#7A8B63",
        "badge_class": "badge-tier-high",
        "description": "AHP Score > 60.0; robust local amenities and connectivity.",
    },
}

METRIC_LABELS: Dict[str, str] = {
    "priority_score": "Composite Priority Score",
    "dim_healthcare_deficit": "Healthcare Access Deficit",
    "dim_demographic_vulnerability": "Demographic Vulnerability",
    "dim_infrastructure_deficit": "Infrastructure Deficit",
    "dim_coastal_exposure": "Coastal Hazard Proximity",
    "vulnerability_archetype_id": "Vulnerability Archetype",
    "preparedness_score": "Authoritative AHP Preparedness Score",
}


def get_custom_css() -> str:
    """Return warm chocolate editorial CSS styles for Streamlit."""
    return """
    <style>
    /* Elegant Editorial Typography */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=DM+Serif+Display:ital@0;1&display=swap');

    /* Global Body and Canvas */
    html, body, .stApp {
        background-color: #211914 !important;
        color: #F3E9DC !important;
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    /* Headings in DM Serif Display */
    h1, h2, h3, .hero-title, .section-heading {
        font-family: 'DM Serif Display', Georgia, 'Times New Roman', serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.015em !important;
        color: #F3E9DC !important;
    }
    h4, h5, h6 {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        color: #D6B28C !important;
    }

    /* Main Container Padding and Max-Width */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3.0rem !important;
        max-width: 94rem !important;
    }

    /* Hero Banner: Warm Editorial Masthead */
    .hero-banner {
        background-color: #2B211B;
        border: 1px solid #514033;
        border-radius: 4px;
        padding: 1.5rem 2.0rem;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.0rem !important;
        line-height: 1.15;
        margin: 0 0 0.4rem 0 !important;
        color: #F3E9DC !important;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #BDAE9C;
        margin: 0;
        line-height: 1.5;
        font-weight: 400;
    }

    /* Metric Cards: Editorial and Tabular */
    .metric-card {
        background-color: #342820;
        border: 1px solid #514033;
        border-radius: 4px;
        padding: 1.0rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #C18B62;
    }
    .metric-title {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #918170;
        margin-bottom: 0.35rem;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .metric-value {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.65rem;
        font-weight: 600;
        color: #C18B62;
        line-height: 1.15;
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum";
    }
    .metric-caption {
        font-size: 0.78rem;
        color: #BDAE9C;
        margin-top: 0.35rem;
        line-height: 1.35;
    }

    /* Restrained Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        border-radius: 2px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-arch-0 { background-color: rgba(122, 139, 99, 0.18); color: #7A8B63; border: 1px solid #7A8B63; }
    .badge-arch-1 { background-color: rgba(212, 139, 71, 0.18); color: #D48B47; border: 1px solid #D48B47; }
    .badge-arch-2 { background-color: rgba(167, 111, 78, 0.18); color: #A76F4E; border: 1px solid #A76F4E; }
    .badge-arch-3 { background-color: rgba(201, 93, 74, 0.18); color: #C95D4A; border: 1px solid #C95D4A; }

    /* Priority Band Badges */
    .badge-band-1 { background-color: rgba(201, 93, 74, 0.18); color: #C95D4A; border: 1px solid #C95D4A; }
    .badge-band-2 { background-color: rgba(212, 139, 71, 0.18); color: #D48B47; border: 1px solid #D48B47; }
    .badge-band-3 { background-color: rgba(168, 142, 117, 0.20); color: #D6B28C; border: 1px solid #A88E75; }
    .badge-band-4 { background-color: rgba(122, 139, 99, 0.18); color: #8D9A78; border: 1px solid #7A8B63; }

    /* AHP Tier Badges */
    .badge-tier-low { background-color: rgba(201, 93, 74, 0.18); color: #C95D4A; border: 1px solid #C95D4A; }
    .badge-tier-medium { background-color: rgba(212, 139, 71, 0.18); color: #D48B47; border: 1px solid #D48B47; }
    .badge-tier-high { background-color: rgba(122, 139, 99, 0.18); color: #8D9A78; border: 1px solid #7A8B63; }

    /* Editorial Callouts & Disclaimers */
    .disclaimer-box {
        background-color: #2B211B;
        border: 1px solid #514033;
        border-left: 3px solid #C18B62;
        border-radius: 3px;
        padding: 0.85rem 1.1rem;
        margin-top: 0.8rem;
        margin-bottom: 1.0rem;
        font-size: 0.83rem;
        color: #BDAE9C;
        line-height: 1.5;
    }
    .disclaimer-box b {
        color: #F3E9DC;
    }

    .data-integrity-warning {
        background-color: #2B211B;
        border: 1px solid #C95D4A;
        border-left: 3px solid #C95D4A;
        border-radius: 3px;
        padding: 0.85rem 1.1rem;
        margin-top: 0.6rem;
        margin-bottom: 1.0rem;
        color: #F3E9DC;
        font-size: 0.85rem;
        line-height: 1.45;
    }

    .filter-banner {
        background-color: #2B211B;
        border: 1px solid #514033;
        border-left: 3px solid #D6B28C;
        border-radius: 3px;
        padding: 0.75rem 1.0rem;
        margin-bottom: 1.0rem;
        font-size: 0.84rem;
        color: #BDAE9C;
    }
    .filter-banner b {
        color: #F3E9DC;
    }

    /* Structured Panels & Checklist Items */
    .checklist-item {
        background-color: #2B211B;
        border: 1px solid #514033;
        border-left: 3px solid #C18B62;
        padding: 0.8rem 1.1rem;
        border-radius: 3px;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: #BDAE9C;
        line-height: 1.5;
    }
    .checklist-item b {
        color: #F3E9DC;
    }

    .summary-card {
        background-color: #342820;
        border: 1px solid #514033;
        border-radius: 4px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.8rem;
    }

    .empty-state {
        text-align: center;
        padding: 2.8rem 1.5rem;
        background-color: #2B211B;
        border: 1px dashed #514033;
        border-radius: 4px;
        margin: 1.5rem 0;
    }

    /* Sidebar: Warm Editorial Dark */
    section[data-testid="stSidebar"] {
        background-color: #2B211B !important;
        border-right: 1px solid #514033 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #F3E9DC !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #514033;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 7px 16px;
        border-radius: 3px;
        background-color: #2B211B;
        border: 1px solid #514033;
        color: #BDAE9C !important;
        font-size: 0.84rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #403127 !important;
        border-color: #C18B62 !important;
        color: #F3E9DC !important;
        font-weight: 600;
    }

    /* Form Inputs and Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #2B211B !important;
        border-color: #514033 !important;
        color: #F3E9DC !important;
        border-radius: 3px !important;
    }
    div[data-baseweb="select"] * {
        color: #F3E9DC !important;
    }
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #BDAE9C !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
    }

    /* Buttons: Restrained Warm Leather Tone */
    .stButton > button {
        background-color: #342820 !important;
        color: #F3E9DC !important;
        border: 1px solid #514033 !important;
        border-radius: 3px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background-color: #403127 !important;
        border-color: #C18B62 !important;
        color: #F3E9DC !important;
    }
    .stButton > button:active {
        background-color: #C18B62 !important;
        color: #211914 !important;
    }

    /* Primary Button */
    .stButton > button[kind="primary"] {
        background-color: #C18B62 !important;
        color: #211914 !important;
        border: 1px solid #D6B28C !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #D6B28C !important;
        border-color: #E0CBB3 !important;
    }

    /* Progress Bars */
    .stProgress > div > div > div > div {
        background-color: #C18B62 !important;
    }
    .stProgress > div > div > div {
        background-color: #342820 !important;
    }

    /* Dividers */
    hr {
        border-color: #514033 !important;
        opacity: 0.7;
        margin: 1.2rem 0 !important;
    }

    /* Dataframe / Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #514033;
        border-radius: 3px;
    }

    /* Tooltip / Details */
    details {
        background-color: #2B211B !important;
        border: 1px solid #514033 !important;
        border-radius: 3px !important;
    }
    summary {
        color: #F3E9DC !important;
        font-weight: 500 !important;
    }
    </style>
    """


def render_header(title: str, subtitle: str, badge: Optional[str] = None) -> str:
    """Generate HTML snippet for warm editorial masthead banners."""
    badge_html = f"<span class='badge-pill' style='background-color:#403127;color:#D6B28C;border:1px solid #514033;margin-left:0.6rem;'>{badge}</span>" if badge else ""
    return f"""
    <div class="hero-banner">
        <h1 class="hero-title">{title} {badge_html}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """


def render_kpi_card(
    title: str,
    value: Any,
    caption: Optional[str] = None,
    percentage: Optional[float] = None,
    color: Optional[str] = None,
    badge_label: Optional[str] = None,
) -> str:
    """Generate HTML snippet for a warm editorial KPI metric card.

    Args:
        title: Metric name or card header.
        value: Primary numerical or text value to display.
        caption: Optional subtitle or contextual footnote.
        percentage: Optional percentage to format beside value.
        color: Optional CSS hex color for value.
        badge_label: Optional small pill badge in top-right of card.

    Returns:
        HTML string suitable for st.markdown(..., unsafe_allow_html=True).
    """
    pct_html = ""
    if percentage is not None:
        pct_html = f" <span style='font-size: 0.88rem; font-weight: 400; color: #918170;'>({percentage:.1f}%)</span>"

    val_color = f"color: {color};" if color else "color: #C18B62;"

    badge_html = ""
    if badge_label:
        badge_html = f"<span style='font-size: 0.70rem; padding: 0.15rem 0.5rem; border-radius: 2px; background: #403127; color: #BDAE9C; border: 1px solid #514033; text-transform: uppercase;'>{badge_label}</span>"

    cap_html = f"<div class='metric-caption'>{caption}</div>" if caption else ""

    return f"""
    <div class="metric-card">
        <div class="metric-title"><span>{title}</span> {badge_html}</div>
        <div class="metric-value" style="{val_color}">{value}{pct_html}</div>
        {cap_html}
    </div>
    """


def render_disclaimer(
    text: str,
    title: str = "Scientific Governance and Baseline Vintage",
) -> str:
    """Generate HTML snippet for scientific disclaimer panels."""
    return f"""
    <div class="disclaimer-box">
        <b>{title}:</b> {text}
    </div>
    """


def render_integrity_alert(text: str) -> str:
    """Generate HTML snippet for data integrity warning banners."""
    return f"""
    <div class="data-integrity-warning">
        <b>Data Integrity Notice:</b> {text}
    </div>
    """


def render_filter_banner(text: str) -> str:
    """Generate HTML snippet for active filter notification banners."""
    return f"""
    <div class="filter-banner">
        <b>Active Scope:</b> {text}
    </div>
    """


def render_empty_state(title: str, message: str) -> str:
    """Generate HTML snippet for empty states."""
    return f"""
    <div class="empty-state">
        <div style="font-family: 'DM Serif Display', Georgia, serif; font-size: 1.25rem; color: #F3E9DC; margin-bottom: 0.35rem;">{title}</div>
        <div style="color: #918170; font-size: 0.85rem;">{message}</div>
    </div>
    """
