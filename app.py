import streamlit as st
from upload import upload_page

# -------------------------------
# Custom CSS for sidebar buttons
# -------------------------------
def sidebar_style():
    st.markdown(
        """
        <style>
        div[data-testid="stSidebar"] button {
            width: 100% !important;
            height: 50px !important;
            margin-bottom: 10px !important;
            font-size: 16px !important;
            font-weight: bold;
            border-radius: 8px !important;
        }
        div[data-testid="stSidebar"] button:focus {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("### ☰ Menu")

    if st.sidebar.button("📤 Upload"):
        st.session_state.page = "upload"
        st.rerun()
    if st.sidebar.button("⚙️ Train"):
        st.session_state.page = "model"
        st.rerun()

# -------------------------------
# Routing
# -------------------------------
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "upload"  # default page

    sidebar_style()

    if st.session_state.page == "upload":
        upload_page()
    elif st.session_state.page == "model":
        st.write("🚀 Training page coming soon!")

if __name__ == "__main__":
    main()