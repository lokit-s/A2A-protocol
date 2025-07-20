import os
import asyncio
import json
import pandas as pd
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import requests
from openai import OpenAI
import streamlit.components.v1 as components

# Initialize OpenAI client with environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🔐 OPENAI_API_KEY environment variable is not set. Please add it to your Streamlit Cloud secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# -------- A2A IMPORTS --------
from python_a2a import AgentNetwork, A2AServer, agent

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
    </style>
""", unsafe_allow_html=True)


# ========== AGENT SETUP ==========
@agent(name="RouterAgent", description="Routes commands to appropriate agents", version="1.0.0")
class RouterAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.network = AgentNetwork(name="Business Management Network")
        self.agent_endpoints = {}
        self.discover_agents()

    def discover_agents(self):
        # Get agent URLs from environment variables with fallbacks
        urls = {
            "ProductAgent": os.environ.get("PRODUCT_AGENT_URL", "http://localhost:5001"),
            "CustomerAgent": os.environ.get("CUSTOMER_AGENT_URL", "http://localhost:5002"),
            "SalesAgent": os.environ.get("SALES_AGENT_URL", "http://localhost:5003")
        }
        
        for agent_name, url in urls.items():
            try:
                self.network.add(agent_name, url)
                self.agent_endpoints[agent_name] = url
                print(f"✅ Added {agent_name} at {url}")
            except Exception as e:
                print(f"❌ Failed to add {agent_name} at {url}: {e}")

    def get_agent_from_llm(self, command):
        system_prompt = """You are an intelligent router for a multi-agent system.
There are three agents: ProductAgent, CustomerAgent, and SalesAgent.
Given a user command, reply with ONLY the name of the agent best suited to handle it.
Reply with 'None' if no agent is suitable.
Examples:
- "Add iPhone to products" -> ProductAgent
- "Add Rahul to customers" -> CustomerAgent
- "Make a sale by customer 1 buys product 2 of quantity 20" -> SalesAgent
- "customer 1 buys 20 of product 1" -> SalesAgent
- "List all sales" -> SalesAgent
- "List all products" -> ProductAgent
- "List all customers" -> CustomerAgent
- "What's the weather?" -> None
"""
        try:
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "20"))
            )
            name = response.choices[0].message.content.strip()
            if name in {"ProductAgent", "CustomerAgent", "SalesAgent"}:
                return name
        except Exception as e:
            print(f"LLM routing error: {e}")
        return None

    def generate_summary(self, response_data, user_query):
        """Generate a natural language summary of the response data"""
        system_prompt = """You are a helpful assistant that generates concise, single-line summaries of database operations.
Given a JSON response from an agent, generate a natural, friendly single-line summary.
Keep it brief and conversational. Don't include the raw data.

Examples:
- For list operations: "Here are your customers/products/sales"
- For add operations: "Successfully added [item name]"
- For delete operations: "Successfully deleted [item name]"
- For update operations: "Successfully updated [item name]"
"""
        try:
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {user_query}\nResponse data: {json.dumps(response_data)}"}
                ],
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.3")),
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "50"))
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Summary generation error: {e}")
            # Fallback summary
            if isinstance(response_data, dict) and response_data.get('status') == 'success':
                return response_data.get('message', 'Operation completed successfully')
            return "Here's your data:"

    def route_and_execute(self, command):
        agent_name = self.get_agent_from_llm(command)
        if not agent_name:
            return {
                'status': 'error',
                'message': 'No agent found',
                'command': command,
                'request_data': {'command': command, 'routed_to': None},
                'response_data': None
            }
        try:
            agent_client = self.network.get_agent(agent_name)

            # Prepare request data for logging
            request_data = {
                'agent': agent_name,
                'endpoint': self.agent_endpoints.get(agent_name, 'Unknown'),
                'command': command,
                'timestamp': pd.Timestamp.now().isoformat()
            }

            # Execute the request
            response = agent_client.ask(command)

            return {
                'status': 'success',
                'routed_to': agent_name,
                'command': command,
                'response': response,
                'request_data': request_data,
                'response_data': response
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Agent error: {e}',
                'command': command,
                'request_data': {
                    'agent': agent_name,
                    'endpoint': self.agent_endpoints.get(agent_name, 'Unknown'),
                    'command': command,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'error': str(e)
                },
                'response_data': {'error': str(e)}
            }


# Initialize router
if "router" not in st.session_state:
    st.session_state.router = RouterAgent()

router = st.session_state.router


# ========== UTILITY FUNCTIONS ==========
def get_image_base64(img_path):
    if os.path.exists(img_path):
        img = Image.open(img_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    return ""


def agent_status_check():
    # Get agent URLs from environment variables
    agent_status = {
        "ProductAgent": os.environ.get("PRODUCT_AGENT_URL", "http://localhost:5001"),
        "CustomerAgent": os.environ.get("CUSTOMER_AGENT_URL", "http://localhost:5002"),
        "SalesAgent": os.environ.get("SALES_AGENT_URL", "http://localhost:5003")
    }
    status_report = {}
    for name, url in agent_status.items():
        try:
            r = requests.get(f"{url}/.well-known/agent.json", timeout=5)
            status_report[name] = {
                "status": "🟢 Online" if r.status_code == 200 else "🔴 Offline",
                "url": url
            }
        except:
            status_report[name] = {
                "status": "🔴 Offline",
                "url": url
            }
    return status_report


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


def parse_response_for_table(response_data):
    """Parse response data to extract tabular information"""
    if not isinstance(response_data, dict):
        return None, None

    # Check for list data in common keys
    for key in ['customers', 'products', 'sales', 'result']:
        if key in response_data and isinstance(response_data[key], list) and response_data[key]:
            return response_data[key], key

    return None, None


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
        llm_options = ["", "GPT-3.5 Turbo", "GPT-4o", "GPT-4"]

        # Auto-select defaults for A2A Application
        protocol_index = protocol_options.index("A2A Protocol") if application == "A2A Business Management" else 0
        llm_index = llm_options.index("GPT-3.5 Turbo") if application == "A2A Business Management" else 0

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
            default_agent = list(st.session_state.available_agents.keys())[
                0] if st.session_state.available_agents else ""
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
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant":
            # Display the LLM summary
            st.markdown(
                f"""
                <div class="chat-row left">
                    <img src="{agent_avatar_url}" class="avatar agent-avatar" alt="Agent">
                    <div class="chat-bubble agent-msg agent-bubble">{msg.get('summary', msg['content'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display table if available
            if 'table_data' in msg and msg['table_data']:
                df = pd.DataFrame(msg['table_data'])
                st.dataframe(df, use_container_width=True)

            # Add detailed output dropdown
            if 'request_data' in msg or 'response_data' in msg:
                with st.expander(f"🔍 Request/Response Details - Message {i + 1}", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**📤 Request Sent to Agent:**")
                        if 'request_data' in msg and msg['request_data']:
                            st.code(json.dumps(msg['request_data'], indent=2), language="json")
                        else:
                            st.code("No request data available", language="text")

                    with col2:
                        st.markdown("**📥 Response Received from Agent:**")
                        if 'response_data' in msg and msg['response_data']:
                            if isinstance(msg['response_data'], (dict, list)):
                                st.code(json.dumps(msg['response_data'], indent=2), language="json")
                            else:
                                st.code(str(msg['response_data']), language="text")
                        else:
                            st.code("No response data available", language="text")

                    # Add metadata
                    if 'agent' in msg:
                        st.markdown(f"**🤖 Routed to Agent:** {msg['agent']}")
                    if 'endpoint' in msg:
                        st.markdown(f"**🌐 Agent Endpoint:** [{msg['endpoint']}]({msg['endpoint']})")

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== CLAUDE-STYLE STICKY CHAT BAR ==========
    st.markdown('<div class="sticky-chatbar"><div class="chatbar-claude">', unsafe_allow_html=True)
    with st.form("chatbar_form", clear_on_submit=True):
        chatbar_cols = st.columns([1, 16, 1])

        # Left: Hamburger (Menu)
        with chatbar_cols[0]:
            hamburger_clicked = st.form_submit_button("≡", use_container_width=True, type="secondary")

        # Middle: Input Box
        with chatbar_cols[1]:
            user_query_input = st.text_input(
                "User Input ",
                placeholder="How can I help you today?",
                label_visibility="collapsed",
                key="chat_input_box"
            )

        # Right: Send Button
        with chatbar_cols[2]:
            send_clicked = st.form_submit_button("➤", use_container_width=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    # ========== FLOATING AGENT MENU ==========
    if st.session_state.get("show_menu", False):
        st.markdown('<div class="tool-menu">', unsafe_allow_html=True)
        st.markdown('<div class="server-title">A2A Business Agents</div>', unsafe_allow_html=True)

        agent_label = "Agents" + (" ▼" if st.session_state["menu_expanded"] else " ▶")
        if st.button(agent_label, key="expand_agents", help="Show agents", use_container_width=True):
            st.session_state["menu_expanded"] = not st.session_state["menu_expanded"]

        if st.session_state["menu_expanded"]:
            st.markdown('<div class="expandable">', unsafe_allow_html=True)
            for agent in st.session_state.agent_states.keys():
                enabled = st.session_state.agent_states[agent]
                new_val = st.toggle(agent, value=enabled, key=f"agent_toggle_{agent}")
                st.session_state.agent_states[agent] = new_val
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== HANDLE HAMBURGER ==========
    if hamburger_clicked:
        st.session_state["show_menu"] = not st.session_state.get("show_menu", False)
        st.rerun()

    # ========== PROCESS CHAT INPUT ==========
    if send_clicked and user_query_input:
        user_query = user_query_input

        try:
            enabled_agents = [k for k, v in st.session_state.agent_states.items() if v]
            if not enabled_agents:
                raise Exception("No agents are enabled. Please enable at least one agent in the menu.")

            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_query,
            })

            # Route and execute via A2A
            result = router.route_and_execute(user_query)

            if result['status'] == 'success':
                # Parse response
                response_data = result.get('response', {})
                routed_agent = result.get('routed_to', 'Unknown')

                # Try to parse JSON if it's a string
                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except json.JSONDecodeError:
                        pass

                # Extract table data
                table_data, table_type = parse_response_for_table(response_data)

                # Generate LLM summary
                summary = router.generate_summary(response_data, user_query)

                assistant_message = {
                    "role": "assistant",
                    "content": str(response_data),  # fallback
                    "summary": summary,
                    "agent": routed_agent,
                    "endpoint": router.agent_endpoints.get(routed_agent, 'Unknown'),
                    "user_query": user_query,
                    "request_data": result.get('request_data', {}),
                    "response_data": result.get('response_data', {})
                }

                # Add table data if available
                if table_data:
                    assistant_message["table_data"] = table_data
                    assistant_message["table_type"] = table_type

            else:
                # Error handling
                assistant_message = {
                    "role": "assistant",
                    "content": f"❌ {result.get('message', 'Routing failed')}",
                    "summary": f"❌ {result.get('message', 'Routing failed')}",
                    "request_data": result.get('request_data', {}),
                    "response_data": result.get('response_data', {})
                }

            st.session_state.messages.append(assistant_message)

        except Exception as e:
            # Exception handling
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Error: {e}",
                "summary": f"⚠️ Error: {e}",
                "request_data": {"error": str(e)},
                "response_data": {"error": str(e)}
            })

        st.rerun()

    # ========== AUTO-SCROLL TO BOTTOM ==========
    components.html("""
        <script>
          setTimeout(() => { window.scrollTo(0, document.body.scrollHeight); }, 80);
        </script>
    """)

# ========== ENHANCED AGENT STATUS FOOTER ==========
with st.expander("🛰️ Agent Network Status", expanded=False):
    st.markdown("**Network:** Business Management Network")
    st.markdown("---")

    status_report = agent_status_check()

    for agent_name, agent_info in status_report.items():
        status = agent_info["status"]
        url = agent_info["url"]

        # Create a styled status display with clickable URL
        st.markdown(f"""
        <div class="agent-status-item">
            <div>
                <strong>{agent_name}:</strong> {status}
            </div>
            <div>
                <a href="{url}" target="_blank" class="agent-url-link">{url}</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Total Agents:** {len(status_report)}")
    online_count = sum(1 for info in status_report.values() if "🟢" in info["status"])
    st.markdown(f"**Online:** {online_count}/{len(status_report)}")

# ========== EXAMPLES & HELP ==========
with st.expander("Examples & Help"):
    st.markdown("""
    ### 📝 Example Commands
    - **Add iPhone with price 999.99 to products**
    - **Add John Doe with email john@example.com to customers**
    - **List all products**
    - **List all customers**
    - **Make a sale by customer 1 buys product 1 of quantity 5**
    - **List all sales**
    - **Delete product 2**
    - **Update customer 1 email to newemail@example.com**
    """)
