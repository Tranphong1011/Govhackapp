import streamlit as st
from database import UserDB
import re


# Initialize database
db = UserDB()


def validate_email(email: str) -> bool:
    """Check email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def show_login_page():
    """Display login page"""
    
    # Page configuration
    st.set_page_config(
        page_title="🔐 Login",
        page_icon="🔐",
        layout="centered"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5rem;
            margin-bottom: 2rem;
        }
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton button {
            width: 100%;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Login System</h1>', unsafe_allow_html=True)
    
    # Tabs for login and registration
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.subheader("Login to the system")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            
            login_btn = st.form_submit_button("🚀 Login")
            
            if login_btn:
                if not username or not password:
                    st.error("❌ Please fill in all information!")
                elif db.verify_user(username, password):
                    # Save login status to session
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_info = db.get_user_info(username)
                    
                    st.success("✅ Login successful!")
                    st.balloons()
                    
                    # Switch to main page
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password!")
    
    with tab2:
        st.subheader("Create new account")
        
        with st.form("register_form"):
            new_username = st.text_input("👤 New Username", placeholder="Username (3-20 characters)")
            new_full_name = st.text_input("👨‍💼 Full Name", placeholder="Enter full name")
            new_email = st.text_input("📧 Email", placeholder="example@email.com")
            new_password = st.text_input("🔒 Password", type="password", placeholder="Password (minimum 6 characters)")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password")
            
            register_btn = st.form_submit_button("📝 Register")
            
            if register_btn:
                # Validate information
                if not all([new_username, new_password, confirm_password]):
                    st.error("❌ Please fill in all required information!")
                elif len(new_username) < 3 or len(new_username) > 20:
                    st.error("❌ Username must be 3-20 characters!")
                elif len(new_password) < 6:
                    st.error("❌ Password must be at least 6 characters!")
                elif new_password != confirm_password:
                    st.error("❌ Password confirmation does not match!")
                elif new_email and not validate_email(new_email):
                    st.error("❌ Invalid email format!")
                else:
                    # Create account
                    if db.create_user(new_username, new_password, new_full_name, new_email):
                        st.success("✅ Registration successful! Please switch to the Login tab.")
                        st.balloons()
                    else:
                        st.error("❌ Username already exists! Please choose a different name.")


def main():
    # Check login status
    if st.session_state.get("logged_in", False):
        # Already logged in, switch to main page
        import main
        main.show_main_page()
    else:
        # Not logged in, display login page
        show_login_page()


if __name__ == "__main__":
    main()
