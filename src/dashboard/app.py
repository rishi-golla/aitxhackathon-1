"""
OSHA Vision Dashboard - Streamlit Application

Main dashboard for factory safety monitoring.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import numpy as np

# Configure page
st.set_page_config(
    page_title="Factory Safety Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
    }
    .violation-card {
        background-color: #2d1f1f;
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .compliant-card {
        background-color: #1f2d1f;
        border-left: 4px solid #4bff4b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .zone-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .zone-ppe { background-color: #ffc107; }
    .zone-hazard { background-color: #ff5722; }
    .zone-restricted { background-color: #f44336; }
    .zone-safe { background-color: #4caf50; }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'violations' not in st.session_state:
    st.session_state.violations = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'frames_processed': 0,
        'violations_detected': 0,
        'alerts_sent': 0,
        'fps': 0.0,
        'compliance_rate': 94.5
    }


def render_sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=OSHA+Vision", width=150)
        st.title("Navigation")

        page = st.radio(
            "Select Page",
            ["📹 Live Monitoring", "🗺️ Zone Configuration", "📊 Reports & Analytics"],
            label_visibility="collapsed"
        )

        st.divider()

        # System Status
        st.subheader("System Status")

        status = "🟢 Active" if st.session_state.pipeline_running else "🔴 Stopped"
        st.markdown(f"**Status:** {status}")

        if st.button("▶️ Start" if not st.session_state.pipeline_running else "⏹️ Stop"):
            st.session_state.pipeline_running = not st.session_state.pipeline_running

        st.divider()

        # Quick Stats
        st.subheader("Quick Stats")
        st.metric("Today's Violations", st.session_state.stats['violations_detected'])
        st.metric("Compliance Rate", f"{st.session_state.stats['compliance_rate']}%")
        st.metric("Current FPS", f"{st.session_state.stats['fps']:.1f}")

        return page


def render_live_monitoring():
    """Render live monitoring page."""
    st.title("📹 Live Monitoring")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Violations Today",
            st.session_state.stats['violations_detected'],
            delta="+2 from yesterday",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "Compliance Rate",
            f"{st.session_state.stats['compliance_rate']}%",
            delta="+1.2%"
        )

    with col3:
        st.metric(
            "Avg Response Time",
            "380ms",
            delta="-20ms"
        )

    with col4:
        st.metric(
            "Active Cameras",
            "4/4",
            delta="All Online"
        )

    st.divider()

    # Main content: Video feeds + Alert feed
    video_col, alert_col = st.columns([2, 1])

    with video_col:
        st.subheader("Camera Feeds")

        # 2x2 camera grid
        cam_row1 = st.columns(2)
        cam_row2 = st.columns(2)

        cameras = [
            ("Camera 1 - Welding Bay", cam_row1[0]),
            ("Camera 2 - Machine Area", cam_row1[1]),
            ("Camera 3 - Chemical Storage", cam_row2[0]),
            ("Camera 4 - General Floor", cam_row2[1])
        ]

        for name, col in cameras:
            with col:
                st.markdown(f"**{name}**")
                # Placeholder for video feed
                placeholder = np.random.randint(0, 50, (240, 320, 3), dtype=np.uint8)
                placeholder[:, :] = [30, 30, 40]  # Dark background
                st.image(placeholder, use_container_width=True)

                # Camera stats
                st.caption("FPS: 30 | Detections: 3")

    with alert_col:
        st.subheader("🚨 Alert Feed")

        # Sample alerts
        alerts = [
            {
                "time": "12:34:56",
                "zone": "Welding Bay A",
                "type": "Missing PPE",
                "missing": ["Face Shield", "Safety Gloves"],
                "severity": "warning",
                "osha": "29 CFR 1910.252"
            },
            {
                "time": "12:30:22",
                "zone": "Machine Area",
                "type": "Missing PPE",
                "missing": ["Hardhat"],
                "severity": "warning",
                "osha": "29 CFR 1910.135"
            },
            {
                "time": "12:25:10",
                "zone": "Chemical Storage",
                "type": "Restricted Entry",
                "missing": [],
                "severity": "critical",
                "osha": "29 CFR 1910.106"
            }
        ]

        for alert in alerts:
            severity_color = "#ff4b4b" if alert["severity"] == "critical" else "#ffc107"
            severity_icon = "🔴" if alert["severity"] == "critical" else "⚠️"

            st.markdown(f"""
            <div class="violation-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>{severity_icon} <strong>{alert['type']}</strong></span>
                    <span style="color: #888;">{alert['time']}</span>
                </div>
                <div style="margin-top: 8px;">
                    <strong>Zone:</strong> {alert['zone']}<br>
                    {"<strong>Missing:</strong> " + ", ".join(alert['missing']) + "<br>" if alert['missing'] else ""}
                    <span style="color: #888; font-size: 12px;">OSHA: {alert['osha']}</span>
                </div>
                <div style="margin-top: 10px;">
                    <button style="background: #4CAF50; border: none; padding: 5px 10px; border-radius: 3px; color: white; margin-right: 5px; cursor: pointer;">✓ Valid</button>
                    <button style="background: #f44336; border: none; padding: 5px 10px; border-radius: 3px; color: white; margin-right: 5px; cursor: pointer;">✗ False</button>
                    <button style="background: #2196F3; border: none; padding: 5px 10px; border-radius: 3px; color: white; cursor: pointer;">🔊 Replay</button>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_zone_configuration():
    """Render zone configuration page."""
    st.title("🗺️ Zone Configuration")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Zone Map")

        # Placeholder for zone editor
        st.info("🖱️ Click and drag to draw zone boundaries on the camera view")

        # Mock zone visualization
        zone_img = np.ones((480, 640, 3), dtype=np.uint8) * 40

        # Draw mock zones
        import cv2
        cv2.rectangle(zone_img, (50, 50), (250, 200), (0, 255, 255), 2)  # Yellow - PPE
        cv2.rectangle(zone_img, (300, 50), (500, 180), (0, 165, 255), 2)  # Orange - Hazard
        cv2.rectangle(zone_img, (50, 250), (200, 400), (0, 0, 255), 2)  # Red - Restricted
        cv2.rectangle(zone_img, (250, 250), (450, 400), (0, 255, 0), 2)  # Green - Safe

        # Labels
        cv2.putText(zone_img, "Welding Bay", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(zone_img, "Machine Area", (310, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(zone_img, "Restricted", (60, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(zone_img, "General Floor", (260, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        st.image(zone_img, channels="BGR", use_container_width=True)

    with col2:
        st.subheader("Zone Settings")

        zone_select = st.selectbox(
            "Select Zone",
            ["Welding Bay A", "Machine Area", "Chemical Storage", "General Floor"]
        )

        st.text_input("Zone Name", value=zone_select)

        zone_type = st.selectbox(
            "Zone Type",
            ["PPE Required", "Hazard", "Restricted", "Safe"]
        )

        st.subheader("Required PPE")

        col_a, col_b = st.columns(2)
        with col_a:
            st.checkbox("Hardhat", value=True)
            st.checkbox("Safety Glasses", value=True)
            st.checkbox("Face Shield")

        with col_b:
            st.checkbox("Safety Vest", value=True)
            st.checkbox("Gloves", value=True)
            st.checkbox("Steel-toe Boots")

        st.text_input("OSHA Reference", value="29 CFR 1910.252")
        st.number_input("Max Occupancy", min_value=1, max_value=50, value=4)

        st.divider()

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("💾 Save Zone", type="primary", use_container_width=True)
        with col_btn2:
            st.button("🗑️ Delete Zone", type="secondary", use_container_width=True)


def render_reports():
    """Render reports and analytics page."""
    st.title("📊 Reports & Analytics")

    # Date range selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    with col3:
        st.selectbox("Zone Filter", ["All Zones", "Welding Bay", "Machine Area", "Chemical Storage"])

    st.divider()

    # Summary metrics
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Violations", "47", delta="-12 vs last week")
    with metric_cols[1]:
        st.metric("Avg Compliance", "94.2%", delta="+2.1%")
    with metric_cols[2]:
        st.metric("False Positives", "3.2%", delta="-0.5%")
    with metric_cols[3]:
        st.metric("Response Time", "340ms", delta="-45ms")

    st.divider()

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Violations Over Time")

        # Mock data for chart
        import pandas as pd
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        violations = np.random.randint(2, 15, len(dates))

        chart_data = pd.DataFrame({
            'Date': dates,
            'Violations': violations
        })

        st.line_chart(chart_data.set_index('Date'))

    with chart_col2:
        st.subheader("Violations by Zone")

        zone_data = pd.DataFrame({
            'Zone': ['Welding Bay', 'Machine Area', 'Chemical Storage', 'General Floor'],
            'Violations': [18, 12, 8, 9]
        })

        st.bar_chart(zone_data.set_index('Zone'))

    # Second row of charts
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.subheader("Violation Types")

        type_data = pd.DataFrame({
            'Type': ['Missing Hardhat', 'Missing Vest', 'Missing Glasses', 'Restricted Entry', 'Overcrowding'],
            'Count': [20, 12, 8, 5, 2]
        })

        st.bar_chart(type_data.set_index('Type'))

    with chart_col4:
        st.subheader("Peak Hours")

        hour_data = pd.DataFrame({
            'Hour': list(range(6, 19)),
            'Violations': [2, 5, 8, 12, 6, 4, 3, 5, 9, 7, 4, 2, 1]
        })

        st.area_chart(hour_data.set_index('Hour'))

    st.divider()

    # Export section
    st.subheader("Export Reports")

    export_cols = st.columns(3)
    with export_cols[0]:
        st.button("📥 Download CSV", use_container_width=True)
    with export_cols[1]:
        st.button("📄 Generate PDF Report", use_container_width=True)
    with export_cols[2]:
        st.button("📧 Email Report", use_container_width=True)


def main():
    """Main dashboard application."""
    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if "Live Monitoring" in page:
        render_live_monitoring()
    elif "Zone Configuration" in page:
        render_zone_configuration()
    elif "Reports" in page:
        render_reports()


if __name__ == "__main__":
    main()
