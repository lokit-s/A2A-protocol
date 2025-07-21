import os
from langchain_groq import ChatGroq
from python_a2a import AgentNetwork, A2AClient, run_server, A2AServer, TaskStatus, TaskState, agent
import json
import threading
import time
import sys

# Initialize Groq client
client = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
)


@agent(
    name="RouterAgent",
    description="Routes commands to appropriate agents in the multi-agent system using Groq AI",
    version="2.0.0"
)
class RouterAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.network = AgentNetwork(name="Business Management Network")
        self.discover_agents()

    def discover_agents(self):
        """Discover available agents using environment variables"""
        agent_urls = {
            "ProductAgent": os.environ.get("PRODUCT_AGENT_URL", "http://localhost:5001"),
            "CustomerAgent": os.environ.get("CUSTOMER_AGENT_URL", "http://localhost:5002"),
            "SalesAgent": os.environ.get("SALES_AGENT_URL", "http://localhost:5003")
        }

        for name, url in agent_urls.items():
            try:
                self.network.add(name, url)
                print(f"✅ Added {name} at {url}")
            except Exception as e:
                print(f"❌ Failed to add {name} at {url}: {e}")

    def get_agent_from_llm(self, command):
        """Use Groq AI to decide which agent should handle the command"""
        system_prompt = """
You are an intelligent router for a multi-agent system.
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
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command}
            ]
            
            response = client.invoke(messages)
            agent_name = response.content.strip()

            if agent_name not in {"ProductAgent", "CustomerAgent", "SalesAgent", "None"}:
                return None
            if agent_name == "None":
                return None
            return agent_name
        except Exception as e:
            print(f"⚠️ Groq routing error: {e}")
            return None

    def route_and_execute(self, command):
        """Route command to appropriate agent and execute"""
        print(f"🔄 RouterAgent: Routing command: '{command}'")
        agent_name = self.get_agent_from_llm(command)

        if not agent_name:
            return {
                'status': 'error',
                'message': 'No suitable agent found for this command',
                'command': command
            }

        try:
            print(f"📍 RouterAgent: Routing to {agent_name}")
            agent_client = self.network.get_agent(agent_name)
            if not agent_client:
                return {
                    'status': 'error',
                    'message': f'Agent {agent_name} not available',
                    'command': command
                }

            response = agent_client.ask(command)
            print(f"✅ RouterAgent: Got response from {agent_name}")

            return {
                'status': 'success',
                'routed_to': agent_name,
                'command': command,
                'response': response
            }

        except Exception as e:
            print(f"❌ RouterAgent: Error communicating with {agent_name}: {e}")
            return {
                'status': 'error',
                'message': f'Error communicating with {agent_name}: {str(e)}',
                'command': command
            }

    def ask(self, message):
        """Handle A2A ask requests - CRITICAL for inter-agent communication"""
        print(f"📞 RouterAgent received ask request: {message}")
        result = self.route_and_execute(message)
        print(f"📤 RouterAgent sending response: {result}")
        return result

    def handle_task(self, task):
        """Handle incoming A2A tasks from Streamlit"""
        print(f"🎯 RouterAgent received task: {task}")
        message_data = task.message or {}
        content = message_data.get("content", {})
        text = content.get("text", "") if isinstance(content, dict) else str(content)

        if not text:
            task.status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message={"role": "agent", "content": {"type": "text",
                                                      "text": "Please provide a command to route to the appropriate agent."}}
            )
            return task

        # Route the command to appropriate agent
        result = self.route_and_execute(text)

        # Format response as artifact
        task.artifacts = [{
            "parts": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }]

        if result['status'] == 'success':
            task.status = TaskStatus(state=TaskState.COMPLETED)
            print(f"✅ RouterAgent: Task completed successfully")
        else:
            task.status = TaskStatus(state=TaskState.FAILED, message=result.get('message'))
            print(f"❌ RouterAgent: Task failed: {result.get('message')}")

        return task

    def interactive_console(self):
        """Interactive console for testing commands"""
        print("\n" + "=" * 60)
        print("🎯 ROUTER AGENT INTERACTIVE CONSOLE (Powered by Groq AI)")
        print("=" * 60)
        print("Available Commands:")
        print("  📦 Product Management:")
        print("    - Add [product name] with price [price] to products")
        print("    - List all products")
        print("    - Delete product ID:[id]")
        print()
        print("  👥 Customer Management:")
        print("    - Add [customer name] to customers")
        print("    - List all customers")
        print("    - Delete customer ID:[id]")
        print()
        print("  💰 Sales Management:")
        print("    - Make a sale by customer [id] buys product [id] of quantity [num]")
        print("    - customer [id] buys [quantity] of product [id]")
        print("    - List all sales")
        print("    - Delete sale ID:[id]")
        print()
        print("  🔧 System Commands:")
        print("    - help, status, quit")
        print("=" * 60)

        while True:
            try:
                command = input("\n🤖 Router (Groq) >>> ").strip()

                if not command:
                    continue

                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break

                if command.lower() == 'help':
                    self.show_help()
                    continue

                if command.lower() == 'status':
                    self.show_status()
                    continue

                print(f"\n🚀 Routing command: '{command}'")
                print("-" * 50)

                agent_name = self.get_agent_from_llm(command)
                if agent_name:
                    print(f"📍 Groq AI routing to: {agent_name}")
                else:
                    print("❓ No suitable agent found by Groq AI")

                result = self.route_and_execute(command)
                self.display_result(result)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Console Error: {e}")

    def show_help(self):
        """Show detailed help"""
        print("\n📚 DETAILED HELP")
        print("-" * 40)
        print("Example Commands:")
        print("  Add iPhone with price 999 to products")
        print("  Add John Doe to customers")
        print("  Make a sale by customer 1 buys product 1 of quantity 5")
        print("  customer 1 buys 20 of product 1")
        print("  List all products")
        print("  List all customers")
        print("  List all sales")

    def show_status(self):
        """Show system status"""
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        print(f"🌐 Network: {self.network.name}")
        print("🧠 AI: Groq (llama3-70b-8192)")
        print("🤖 Available Agents:")

        agent_status = {
            "ProductAgent": os.environ.get("PRODUCT_AGENT_URL", "http://localhost:5001"),
            "CustomerAgent": os.environ.get("CUSTOMER_AGENT_URL", "http://localhost:5002"),
            "SalesAgent": os.environ.get("SALES_AGENT_URL", "http://localhost:5003")
        }

        for agent_name, url in agent_status.items():
            try:
                import requests
                response = requests.get(f"{url}/.well-known/agent.json", timeout=2)
                status = "🟢 Online" if response.status_code == 200 else "🔴 Offline"
            except:
                status = "🔴 Offline"

            print(f"  - {agent_name}: {status} ({url})")

    def display_result(self, result):
        """Display command result in a formatted way"""
        print("\n📋 RESULT:")
        print("-" * 30)

        if result['status'] == 'success':
            print("✅ Status: SUCCESS")
            print(f"🎯 Routed to: {result.get('routed_to', 'Unknown')}")

            response = result.get('response', {})
            if isinstance(response, dict):
                if 'status' in response and response['status'] == 'success':
                    print(f"💬 Message: {response.get('message', 'No message')}")

                    action = response.get('action', '')
                    if action == 'add_product' and 'product' in response:
                        product = response['product']
                        price = product.get('price', 'N/A')
                        print(f"📦 Product Added: ID={product['id']}, Name='{product['name']}', Price=${price}")
                    elif action == 'add_customer' and 'customer' in response:
                        customer = response['customer']
                        print(f"👤 Customer Added: ID={customer['id']}, Name='{customer['name']}'")
                    elif action == 'make_sale' and 'sale' in response:
                        sale = response['sale']
                        total = sale.get('total_price', 'N/A')
                        print(
                            f"💰 Sale Made: ID={sale['id']}, {sale['customer_name']} bought {sale['quantity']}x {sale['product_name']} = ${total}")
                    elif action.startswith('list_'):
                        items = response.get(action.split('_')[1], [])
                        print(f"📋 Found {len(items)} items")
                        for item in items[:3]:
                            if 'name' in item:
                                if 'price' in item:
                                    print(f"   - ID:{item['id']} {item['name']} (${item['price']})")
                                else:
                                    print(f"   - ID:{item['id']} {item['name']}")
                else:
                    print(f"❌ Agent Error: {response.get('message', 'Unknown error')}")
            else:
                print(f"📄 Raw Response: {response}")
        else:
            print("❌ Status: FAILED")
            print(f"💬 Error: {result.get('message', 'Unknown error')}")

    def run_with_console(self):
        """Run the server and console in parallel"""
        port = int(os.environ.get("PORT", 5000))
        server_thread = threading.Thread(
            target=lambda: run_server(self, host='0.0.0.0', port=port),
            daemon=True
        )
        server_thread.start()

        time.sleep(2)
        print(f"🚀 Router Agent server running on port {port}")

        self.interactive_console()


if __name__ == '__main__':
    print("🌐 Starting Router Agent with Groq AI-powered routing...")
    router = RouterAgent()

    print(f"\n📋 Available agents in network:")
    try:
        for agent_info in router.network.list_agents():
            print(f"  - {agent_info.get('name', 'Unknown')}")
    except:
        print("  - ProductAgent")
        print("  - CustomerAgent") 
        print("  - SalesAgent")

    # Get port from environment variable for cloud deployment
    port = int(os.environ.get("PORT", 5000))
    
    if len(sys.argv) > 1 and sys.argv[1] == '--server-only':
        run_server(router, host='0.0.0.0', port=port)
        print(f"🚀 Router Agent running on port {port}")
    else:
        router.run_with_console()
