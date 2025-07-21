import os
import json
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import requests
import streamlit.components.v1 as components
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Groq client with environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🔐 GROQ_API_KEY environment variable is not set. Please add it to your Streamlit Cloud secrets.")
    st.stop()

client = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
)

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="A2A Business Management", layout="wide")

# ========== GLOBAL CSS ==========
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4286f4 0%, #397dd2 100%);
        color: #fff !important;
        min-width: 330px !important;
        padding: 0 0 0 0 !important;
    }
    [data-testid="stSidebar"] .sidebar-title {
        color: #fff !important;
        font-weight: bold;
        font-size: 2.2rem;
        letter-spacing: -1px;
        text-align: center;
        margin-top: 28px;
        margin-bottom: 18px;
    }
    .sidebar-block {
        width: 94%;
        margin: 0 auto 18px auto;
    }
    .sidebar-block label {
        color: #fff !important;
        font-weight: 500;
        font-size: 1.07rem;
        margin-bottom: 4px;
        margin-left: 2px;
        display: block;
        text-align: left;
    }
    .sidebar-block .stSelectbox>div {
        background: #fff !important;
        color: #222 !important;
        border-radius: 13px !important;
        font-size: 1.13rem !important;
        min-height: 49px !important;
        box-shadow: 0 3px 14px #0002 !important;
        padding: 3px 10px !important;
        margin-top: 4px !important;
        margin-bottom: 0 !important;
    }
    .stButton>button {
            width: 100%;
            height: 3rem;
            background: #39e639;
            color: #222;
            font-size: 1.1rem;
            font-weight: bold;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
    /* Small refresh button styling */
    .small-refresh-button button {
        width: auto !important;
        height: 2rem !important;
        background: #4286f4 !important;
        color: #fff !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        margin-bottom: 0.5rem !important;
        padding: 0.25rem 0.75rem !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .small-refresh-button button:hover {
        background: #397dd2 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    .sidebar-logo-label {
        margin-top: 30px !important;
        margin-bottom: 10px;
        font-size: 1.13rem !important;
        font-weight: 600;
        text-align: center;
        color: #fff !important;
        letter-spacing: 0.1px;
    }
    .sidebar-logo-row {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
    .sidebar-logo-row img {
        width: 42px;
        height: 42px;
        border-radius: 9px;
        background: #fff;
        padding: 6px 8px;
        object-fit: contain;
        box-shadow: 0 2px 8px #0002;
    }
    /* Chat area needs bottom padding so sticky bar does not overlap */
    .stChatPaddingBottom { padding-bottom: 98px; }
    /* Responsive sticky chatbar */
    .sticky-chatbar {
        position: fixed;
        left: 330px;
        right: 0;
        bottom: 0;
        z-index: 100;
        background: #f8fafc;
        padding: 0.6rem 2rem 0.8rem 2rem;
        box-shadow: 0 -2px 24px #0001;
    }
    @media (max-width: 800px) {
        .sticky-chatbar { left: 0; right: 0; padding: 0.6rem 0.5rem 0.8rem 0.5rem; }
        [data-testid="stSidebar"] { display: none !important; }
    }
    .chat-bubble {
        padding: 13px 20px;
        margin: 8px 0;
        border-radius: 18px;
        max-width: 75%;
        font-size: 1.09rem;
        line-height: 1.45;
        box-shadow: 0 1px 4px #0001;
        display: inline-block;
        word-break: break-word;
    }
    .user-msg {
        background: #e6f0ff;
        color: #222;
        margin-left: 24%;
        text-align: right;
        border-bottom-right-radius: 6px;
        border-top-right-radius: 24px;
    }
    .agent-msg {
        background: #f5f5f5;
        color: #222;
        margin-right: 24%;
        text-align: left;
        border-bottom-left-radius: 6px;
        border-top-left-radius: 24px;
    }
    .chat-row {
        display: flex;
        align-items: flex-end;
        margin-bottom: 0.6rem;
    }
    .avatar {
        height: 36px;
        width: 36px;
        border-radius: 50%;
        margin: 0 8px;
        object-fit: cover;
        box-shadow: 0 1px 4px #0002;
    }
    .user-avatar { order: 2; }
    .agent-avatar { order: 0; }
    .user-bubble { order: 1; }
    .agent-bubble { order: 1; }
    .right { justify-content: flex-end; }
    .left { justify-content: flex-start; }
    .chatbar-claude {
        display: flex;
        align-items: center;
        gap: 12px;
        width: 100%;
        max-width: 850px;
        margin: 0 auto;
        border-radius: 20px;
        background: #fff;
        box-shadow: 0 2px 8px #0002;
        padding: 8px 14px;
        margin-bottom: 0;
    }
    .claude-hamburger {
        background: #f2f4f9;
        border: none;
        border-radius: 11px;
        font-size: 1.35rem;
        font-weight: bold;
        width: 38px; height: 38px;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.13s;
    }
    .claude-hamburger:hover { background: #e6f0ff; }
    .claude-input {
        flex: 1;
        border: none;
        outline: none;
        font-size: 1.12rem;
        padding: 0.45rem 0.5rem;
        background: #f5f7fa;
        border-radius: 8px;
        min-width: 60px;
    }
    .claude-send {
        background: #fe3044 !important;
        color: #fff !important;
        border: none;
        border-radius: 50%;
        width: 40px; height: 40px;
        font-size: 1.4rem !important;
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: background 0.17s;
    }
    .claude-send:hover { background: #d91d32 !important; }
    .tool-menu {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        min-width: 200px;
    }
    .server-title {
        font-weight: bold;
        margin-bottom: 10px;
        color: #333;
    }
    .expandable {
        margin-top: 8px;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #fff !important;
        font-weight: 500;
        font-size: 1.07rem;
    }
    .agent-link {
        color: #4286f4;
        text-decoration: none;
        font-size: 0.9rem;
        margin-left: 8px;
    }
    .agent-link:hover {
        text-decoration: underline;
    }
    .agent-info {
        display: flex;
        align-items: center;
        margin: 4px 0;
    }
    .agent-status-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    .agent-status-item:last-child {
        border-bottom: none;
    }
    .agent-url-link {
        color: #4286f4;
        text-decoration: none;
        font-size: 0.85rem;
        padding: 2px 6px;
        background: #f0f8ff;
        border-radius: 4px;
        margin-left: 8px;
    }
    .agent-url-link:hover {
        background: #e6f0ff;
        text-decoration: underline;
    }
    .connection-status {
        padding: 8px 12px;
        border-radius: 6px;
        margin: 10px 0;
        font-weight: 500;
    }
    .connection-online {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .connection-offline {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .data-summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .data-summary-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .data-summary-stats {
        display: flex;
        gap: 20px;
        margin-top: 10px;
    }
    .data-stat {
        text-align: center;
    }
    .data-stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        display: block;
    }
    .data-stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        display: block;
    }
    .pretty-table {
        background: white;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========== UTILITY FUNCTIONS ==========
def get_image_base64(img_path):
    if os.path.exists(img_path):
        img = Image.open(img_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return ""

def check_router_connection():
    """Check if RouterAgent is available"""
    router_url = os.environ.get("ROUTER_AGENT_URL", "http://localhost:5000")
    try:
        response = requests.get(f"{router_url}/.well-known/agent.json", timeout=5)
        return response.status_code == 200, router_url
    except:
        return False, router_url

def send_command_to_router(command):
    """Send command to RouterAgent via A2A protocol"""
    router_url = os.environ.get("ROUTER_AGENT_URL", "http://localhost:5000")
    
    try:
        # Use A2A protocol to send command
        payload = {
            "message": {
                "content": {
                    "text": command
                }
            }
        }
        
        response = requests.post(f"{router_url}/tasks/send", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return {
            'status': 'success',
            'response': result,
            'endpoint': router_url
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'message': f'Router connection error: {str(e)}',
            'endpoint': router_url
        }
    except Exception as e:
        return {
            'status': 'error', 
            'message': f'Unexpected error: {str(e)}',
            'endpoint': router_url
        }

def agent_status_check():
    """Check status of all agents through RouterAgent"""
    agent_status = {
        "RouterAgent": os.environ.get("ROUTER_AGENT_URL", "http://localhost:5000"),
        "ProductAgent": os.environ.get("PRODUCT_AGENT_URL", "http://localhost:5001"),
        "CustomerAgent": os.environ.get("CUSTOMER_AGENT_URL", "http://localhost:5002"),
        "SalesAgent": os.environ.get("SALES_AGENT_URL", "http://localhost:5003")
    }
    
    status_report = {}
    for name, url in agent_status.items():
        try:
            response = requests.get(f"{url}/.well-known/agent.json", timeout=5)
            status_report[name] = {
                "status": "🟢 Online" if response.status_code == 200 else "🔴 Offline",
                "url": url
            }
        except:
            status_report[name] = {
                "status": "🔴 Offline", 
                "url": url
            }
    return status_report

def generate_summary(response_data, user_query):
    """Generate a natural language summary using Groq"""
    system_prompt = """You are a helpful assistant that generates concise, single-line summaries of database operations.
Given a JSON response from an agent, generate a natural, friendly single-line summary.
Keep it brief and conversational. Don't include the raw data.

Examples:
- For list operations: "Here are your customers/products/sales"
- For add operations: "Successfully added [item name]"
- For delete operations: "Successfully deleted [item name]"
- For update operations: "Successfully updated [item name]"
- For sales: "Sale recorded successfully with total amount"
"""
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {user_query}\nResponse data: {json.dumps(response_data)}")
        ]
        response = client.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
        # Fallback summary
        if isinstance(response_data, dict) and response_data.get('status') == 'success':
            return response_data.get('message', 'Operation completed successfully')
        return "Here's your data:"

def parse_nested_response(router_response_text):
    """Parse nested JSON response from RouterAgent"""
    try:
        # First, parse the outer RouterAgent response
        outer_response = json.loads(router_response_text)
        
        # Extract the inner agent response
        if 'response' in outer_response:
            inner_response_text = outer_response['response']
            if isinstance(inner_response_text, str):
                # Parse the inner JSON string
                inner_response = json.loads(inner_response_text)
                
                # Add router metadata to the response
                inner_response['_router_info'] = {
                    'routed_to': outer_response.get('routed_to'),
                    'command': outer_response.get('command'),
                    'status': outer_response.get('status')
                }
                
                return inner_response
        
        return outer_response
        
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": router_response_text}

def create_data_summary_card(data, data_type):
    """Create a beautiful summary card for data"""
    if not data or not isinstance(data, list):
        return None
    
    count = len(data)
    
    # Get appropriate emoji and stats based on data type
    if data_type == 'sales':
        emoji = "💰"
        title = "Sales Summary"
        total_amount = sum(float(item.get('total_price', 0)) for item in data if 'total_price' in item)
        avg_amount = total_amount / count if count > 0 else 0
        stats_html = f"""
        <div class="data-summary-stats">
            <div class="data-stat">
                <span class="data-stat-number">{count}</span>
                <span class="data-stat-label">Total Sales</span>
            </div>
            <div class="data-stat">
                <span class="data-stat-number">${total_amount:,.2f}</span>
                <span class="data-stat-label">Revenue</span>
            </div>
            <div class="data-stat">
                <span class="data-stat-number">${avg_amount:,.2f}</span>
                <span class="data-stat-label">Avg Sale</span>
            </div>
        </div>
        """
    elif data_type == 'products':
        emoji = "📦"
        title = "Products Summary"
        avg_price = sum(float(item.get('price', 0)) for item in data if 'price' in item) / count if count > 0 else 0
        stats_html = f"""
        <div class="data-summary-stats">
            <div class="data-stat">
                <span class="data-stat-number">{count}</span>
                <span class="data-stat-label">Total Products</span>
            </div>
            <div class="data-stat">
                <span class="data-stat-number">${avg_price:,.2f}</span>
                <span class="data-stat-label">Avg Price</span>
            </div>
        </div>
        """
    elif data_type == 'customers':
        emoji = "👥"
        title = "Customers Summary"
        with_email = sum(1 for item in data if item.get('email'))
        stats_html = f"""
        <div class="data-summary-stats">
            <div class="data-stat">
                <span class="data-stat-number">{count}</span>
                <span class="data-stat-label">Total Customers</span>
            </div>
            <div class="data-stat">
                <span class="data-stat-number">{with_email}</span>
                <span class="data-stat-label">With Email</span>
            </div>
        </div>
        """
    else:
        emoji = "📊"
        title = f"{data_type.title()} Summary"
        stats_html = f"""
        <div class="data-summary-stats">
            <div class="data-stat">
                <span class="data-stat-number">{count}</span>
                <span class="data-stat-label">Total Items</span>
            </div>
        </div>
        """
    
    return f"""
    <div class="data-summary-card">
        <div class="data-summary-title">{emoji} {title}</div>
        {stats_html}
    </div>
    """

def format_dataframe_for_display(df, data_type):
    """Format dataframe with better column names and styling"""
    df_display = df.copy()
    
    # Format specific columns based on data type
    if data_type == 'sales':
        # Format price columns
        if 'price' in df_display.columns:
            df_display['price'] = df_display['price'].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "")
        if 'total_price' in df_display.columns:
            df_display['total_price'] = df_display['total_price'].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "")
        
        # Rename columns for better display
        column_renames = {
            'id': 'Sale ID',
            'customer_id': 'Customer ID',
            'customer_name': 'Customer Name',
            'product_id': 'Product ID', 
            'product_name': 'Product Name',
            'quantity': 'Qty',
            'price': 'Unit Price',
            'total_price': 'Total Price',
            'sale_time': 'Sale Date'
        }
        
    elif data_type == 'products':
        # Format price column
        if 'price' in df_display.columns:
            df_display['price'] = df_display['price'].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "")
        
        column_renames = {
            'id': 'Product ID',
            'name': 'Product Name',
            'description': 'Description',
            'price': 'Price',
            'created_at': 'Created Date'
        }
        
    elif data_type == 'customers':
        column_renames = {
            'id': 'Customer ID',
            'name': 'Customer Name',
            'email': 'Email Address',
            'created_at': 'Created Date'
        }
    else:
        column_renames = {}
    
    # Apply column renames
    for old_name, new_name in column_renames.items():
        if old_name in df_display.columns:
            df_display.rename(columns={old_name: new_name}, inplace=True)
    
    return df_display

def parse_response_for_table(response_data):
    """Parse response data to extract tabular information"""
    if not isinstance(response_data, dict):
        return None, None

    # Check for list data in common keys
    for key in ['sales', 'products', 'customers', 'result']:
        if key in response_data and isinstance(response_data[key], list) and response_data[key]:
            return response_data[key], key

    return None, None

def discover_agents():
    """Discover available A2A agents with their endpoints"""
    available_agents = {
        "ProductAgent": {
            "description": "Manages product inventory, pricing, and catalog operations",
            "endpoint": os.environ.get("PRODUCT_AGENT_URL", "http://localhost:5001")
        },
        "CustomerAgent": {
            "description": "Handles customer data, profiles, and relationship management", 
            "endpoint": os.environ.get("CUSTOMER_AGENT_URL", "http://localhost:5002")
        },
        "SalesAgent": {
            "description": "Processes sales transactions, orders, and revenue analytics",
            "endpoint": os.environ.get("SALES_AGENT_URL", "http://localhost:5003")
        }
    }
    return available_agents

# ========== SIDEBAR NAVIGATION ==========
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Solutions Scope</div>", unsafe_allow_html=True)
    with st.container():
        # Application selectbox
        application = st.selectbox(
            "Select Application",
            ["Select Application", "A2A Business Management"],
            key="app_select"
        )

        # Dynamic options based on application selection
        protocol_options = ["", "A2A Protocol"]
        llm_options = ["", "Groq Llama3-70B", "Groq Llama3-8B", "Groq Mixtral-8x7B"]

        # Auto-select defaults for A2A Application
        protocol_index = protocol_options.index("A2A Protocol") if application == "A2A Business Management" else 0
        llm_index = llm_options.index("Groq Llama3-70B") if application == "A2A Business Management" else 0

        protocol = st.selectbox(
            "Protocol",
            protocol_options,
            key="protocol_select",
            index=protocol_index
        )

        llm_model = st.selectbox(
            "LLM Models",
            llm_options,
            key="llm_select",
            index=llm_index
        )

        # Dynamic server agents selection
        if application == "A2A Business Management" and "available_agents" in st.session_state:
            agent_options = [""] + list(st.session_state.available_agents.keys())
            default_agent = list(st.session_state.available_agents.keys())[0] if st.session_state.available_agents else ""
            agent_index = agent_options.index(default_agent) if default_agent else 0
        else:
            agent_options = ["", "ProductAgent", "CustomerAgent", "SalesAgent"]
            agent_index = 0

        server_agents = st.selectbox(
            "Agents",
            agent_options,
            key="server_agents",
            index=agent_index
        )

        if st.button("Clear/Reset", key="clear_button"):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="sidebar-logo-label">Build & Deployed on</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-logo-row">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/googlecloud/googlecloud-original.svg" title="Google Cloud">
            <img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" title="AWS">
            <img src="https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg" title="Azure Cloud">
        </div>
        """,
        unsafe_allow_html=True
    )

# ========== LOGO/HEADER FOR MAIN AREA ==========
logo_path = "Logo.png"
logo_base64 = get_image_base64(logo_path) if os.path.exists(logo_path) else ""
if logo_base64:
    st.markdown(
        f"""
        <div style='display: flex; flex-direction: column; align-items: center; margin-bottom:20px;'>
            <img src='data:image/png;base64,{logo_base64}' width='220'>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 18px;
        padding: 10px 0 10px 0;
    ">
        <span style="
            font-size: 2.5rem;
            font-weight: bold;
            letter-spacing: -2px;
            color: #222;
        ">
            A2A-Driven Business Management Platform
        </span>
        <span style="
            font-size: 1.15rem;
            color: #555;
            margin-top: 0.35rem;
        ">
            Agentic Platform: Leveraging A2A Protocol and LLMs for Intelligent Business Operations and Real-time Analytics.
        </span>
        <hr style="
        width: 80%;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4286f4, transparent);
        margin: 20px auto;
        ">
    </div>
    """,
    unsafe_allow_html=True
)

# ========== SESSION STATE INITIALIZATION ==========
if "messages" not in st.session_state:
    st.session_state.messages = []

if "available_agents" not in st.session_state:
    st.session_state.available_agents = {}

if "agent_states" not in st.session_state:
    st.session_state.agent_states = {}

if "show_menu" not in st.session_state:
    st.session_state["show_menu"] = False

if "menu_expanded" not in st.session_state:
    st.session_state["menu_expanded"] = True

# ========== MAIN APPLICATION LOGIC ==========
if application == "A2A Business Management":
    user_avatar_url = "https://cdn-icons-png.flaticon.com/512/1946/1946429.png"
    agent_avatar_url = "https://cdn-icons-png.flaticon.com/512/4712/4712039.png"

    # Check RouterAgent connection
    router_online, router_url = check_router_connection()
    
    if router_online:
        st.markdown(f"""
        <div class="connection-status connection-online">
            🟢 Connected to RouterAgent at {router_url}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="connection-status connection-offline">
            🔴 RouterAgent offline at {router_url}
        </div>
        """, unsafe_allow_html=True)

    # Discover agents dynamically if not already done
    if not st.session_state.available_agents:
        discovered_agents = discover_agents()
        st.session_state.available_agents = discovered_agents
        st.session_state.agent_states = {agent: True for agent in discovered_agents.keys()}

    # ========== RENDER CHAT MESSAGES ==========
    st.markdown('<div class="stChatPaddingBottom">', unsafe_allow_html=True)

    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-row right">
                    <div class="chat-bubble user-msg user-bubble">{msg['content']}</div>
                    <img src="{user_avatar_url}" class="avatar user-avatar" alt="User">
                </div
