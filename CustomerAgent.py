import psycopg2
import psycopg2.extras
import json
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState
import os

# Initialize Groq client with environment variable
client = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
)

# Database configuration from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

@agent(
    name="CustomerAgent",
    description="Manages customer database operations using natural language",
    version="1.2.0"
)
class CustomerAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.conn = None
        self.cursor = None
        self.init_database()
        print("👥 CustomerAgent PostgreSQL database initialized")

    def init_database(self):
        """Initialize PostgreSQL connection and create table if needed"""
        try:
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL environment variable is not set")
            
            self.conn = psycopg2.connect(DATABASE_URL)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Create customers table if it doesn't exist
            self.cursor.execute('''
                                CREATE TABLE IF NOT EXISTS customers
                                (
                                    id
                                    SERIAL
                                    PRIMARY
                                    KEY,
                                    name
                                    VARCHAR
                                (
                                    255
                                ) NOT NULL,
                                    email VARCHAR
                                (
                                    255
                                ),
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                    )
                                ''')
            self.conn.commit()
            print("✅ PostgreSQL customers table ready")
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            raise

    def reconnect_if_needed(self):
        """Reconnect to database if connection is lost"""
        try:
            if self.conn and not self.conn.closed:
                self.cursor.execute('SELECT 1')
                return
        except:
            pass

        print("🔄 Reconnecting to database...")
        self.init_database()

    def add_customer(self, name, email=None):
        """Add a new customer to the database"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute(
                'INSERT INTO customers (name, email) VALUES (%s, %s) RETURNING id',
                (name, email)
            )
            customer_id = self.cursor.fetchone()['id']
            self.conn.commit()
            return customer_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Add customer error: {e}")
            raise

    def list_customers(self):
        """List all customers from the database"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('SELECT id, name, email, created_at FROM customers ORDER BY id')
            rows = self.cursor.fetchall()
            return [(row['id'], row['name'], row['email'], str(row['created_at'])) for row in rows]
        except Exception as e:
            print(f"❌ List customers error: {e}")
            raise

    def get_customer(self, customer_id):
        """Get a specific customer by ID"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('SELECT id, name, email, created_at FROM customers WHERE id = %s', (customer_id,))
            row = self.cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'email': row['email'],
                    'created_at': str(row['created_at'])
                }
            return None
        except Exception as e:
            print(f"❌ Get customer error: {e}")
            raise

    def delete_customer(self, customer_id):
        """Delete a customer by ID"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('DELETE FROM customers WHERE id = %s', (customer_id,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            return deleted_count
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Delete customer error: {e}")
            raise

    def update_customer(self, customer_id, name=None, email=None):
        """Update a customer's information"""
        self.reconnect_if_needed()
        try:
            updates = []
            params = []

            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if email is not None:
                updates.append("email = %s")
                params.append(email)

            if not updates:
                return 0

            params.append(customer_id)
            sql = f'UPDATE customers SET {", ".join(updates)} WHERE id = %s'

            self.cursor.execute(sql, params)
            updated_count = self.cursor.rowcount
            self.conn.commit()
            return updated_count
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Update customer error: {e}")
            raise

    def process_customer_command(self, command: str) -> dict:
        """Process natural language commands for customer management"""
        print(f"🔍 CustomerAgent received command: {command}")

        system_prompt = """
You are an assistant that converts user requests about customers into structured JSON commands.
Supported commands:
- To add a customer: {"intent":"add_customer","parameters":{"name":"customer name","email":"optional email"}}
- To list customers: {"intent":"list_customers","parameters":{}}
- To get a customer: {"intent":"get_customer","parameters":{"id": customer_id}}
- To delete a customer: {"intent":"delete_customer","parameters":{"id": customer_id}}
- To update a customer: {"intent":"update_customer","parameters":{"id": customer_id, "name": "new name (optional)", "email": "new email (optional)"}}
Return only the JSON, no extra text.
"""
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=command)
            ]
            response = client.invoke(messages)
            result = response.content.strip()
            print(f"🤖 CustomerAgent LLM response: {result}")

            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\{.*\}', result, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                else:
                    raise ValueError("No JSON found in LLM response.")

            intent = parsed.get("intent")
            params = parsed.get("parameters", {})

            if intent == "add_customer":
                name = params.get("name", "").strip()
                if not name:
                    raise ValueError("Customer name missing")
                email = params.get("email", None)
                customer_id = self.add_customer(name, email)
                return {
                    'status': 'success',
                    'action': 'add_customer',
                    'message': f'Customer "{name}" added',
                    'customer': {'id': customer_id, 'name': name, 'email': email}
                }

            elif intent == "list_customers":
                customers = self.list_customers()
                formatted_customers = [
                    {
                        'id': c[0],
                        'name': c[1],
                        'email': c[2],
                        'created_at': c[3]
                    } for c in customers
                ]
                return {
                    'status': 'success',
                    'action': 'list_customers',
                    'message': f'Found {len(customers)} customer(s)',
                    'customers': formatted_customers,
                    'count': len(customers)
                }

            elif intent == "get_customer":
                cust_id = params.get("id")
                if not cust_id:
                    raise ValueError("Customer ID missing")
                customer = self.get_customer(cust_id)
                print(f"🔍 CustomerAgent found customer: {customer}")
                if customer:
                    return {
                        'status': 'success',
                        'action': 'get_customer',
                        'message': f'Customer {cust_id} found',
                        'customer': customer
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'get_customer',
                        'message': f'No customer found with ID {cust_id}'
                    }

            elif intent == "delete_customer":
                cust_id = params.get("id")
                if not cust_id:
                    raise ValueError("Customer ID missing")
                deleted = self.delete_customer(cust_id)
                if deleted:
                    return {
                        'status': 'success',
                        'action': 'delete_customer',
                        'message': f'Customer with ID {cust_id} deleted'
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'delete_customer',
                        'message': f'No customer found with ID {cust_id}'
                    }

            elif intent == "update_customer":
                cust_id = params.get("id")
                if not cust_id:
                    raise ValueError("Customer ID missing")
                name = params.get("name", None)
                email = params.get("email", None)
                updated = self.update_customer(cust_id, name, email)
                if updated:
                    return {
                        'status': 'success',
                        'action': 'update_customer',
                        'message': f'Customer with ID {cust_id} updated'
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'update_customer',
                        'message': f'No customer found with ID {cust_id} or nothing to update'
                    }

            else:
                return {
                    'status': 'error',
                    'action': 'unknown',
                    'message': 'Command not recognized'
                }

        except Exception as e:
            print(f"❌ CustomerAgent error: {e}")
            return {
                'status': 'error',
                'action': 'parse_command',
                'message': f'Command failed: {str(e)}'
            }

    def ask(self, message):
        """Handle A2A ask requests - CRITICAL for inter-agent communication"""
        print(f"📞 CustomerAgent received ask request: {message}")
        result = self.process_customer_command(message)
        print(f"📤 CustomerAgent sending response: {result}")
        return result

    @skill(
        name="manage_customers",
        description="Add, update, list, and delete customer records via RouterAgent",
        examples=[
            "Add John Doe to customers",
            "Add customer with name Sarah Smith and email sarah@example.com",
            "List all customers",
            "Get customer 1",
            "Update customer 2 name to Michael Johnson",
            "Delete customer 3"
        ]
    )
    def handle_task(self, task):
        """Handle incoming A2A tasks"""
        message_data = task.message or {}
        content = message_data.get("content", {})
        text = content.get("text", "") if isinstance(content, dict) else str(content)

        if not text:
            task.status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message={"role": "agent", "content": {"type": "text",
                                                      "text": "Please provide a customer management command."}}
            )
            return task

        result = self.process_customer_command(text)

        task.artifacts = [{
            "parts": [{"type": "text", "text": json.dumps(result, indent=2)}]
        }]

        if result['status'] == 'success':
            task.status = TaskStatus(state=TaskState.COMPLETED)
        else:
            task.status = TaskStatus(state=TaskState.FAILED, message=result.get('message'))

        return task

    def __del__(self):
        """Clean up database connections"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
        except:
            pass


if __name__ == '__main__':
    # Get port from environment (Render automatically sets PORT)
    port = int(os.environ.get('PORT', 5002))
    host = os.environ.get('HOST', '0.0.0.0')  # Render requires 0.0.0.0
    
    agent = CustomerAgent()
    print(f"🚀 Customer Agent running on {host}:{port}")
    run_server(agent, host=host, port=port)
