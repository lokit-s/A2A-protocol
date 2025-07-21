import json
from datetime import datetime
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_groq import ChatGroq
from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState

# Initialize Groq client
client = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
)


@agent(
    name="CustomerAgent",
    description="Manages customer database operations using natural language with PostgreSQL",
    version="3.0.0"
)
class CustomerAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        self.init_database()
        print("👤 CustomerAgent database initialized with PostgreSQL")

    def get_connection(self):
        """Get a new database connection"""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def init_database(self):
        """Initialize the database schema"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS customers (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        email VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                print("✅ Customers table created/verified")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def add_customer(self, name, email=None):
        """Add a new customer"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO customers (name, email) VALUES (%s, %s) RETURNING id',
                    (name, email)
                )
                customer_id = cursor.fetchone()['id']
                conn.commit()
                return customer_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def list_customers(self):
        """List all customers"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, name, email, created_at FROM customers ORDER BY id')
                return cursor.fetchall()
        finally:
            conn.close()

    def get_customer(self, customer_id):
        """Get a specific customer"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, name, email, created_at FROM customers WHERE id = %s', (customer_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        finally:
            conn.close()

    def delete_customer(self, customer_id):
        """Delete a customer"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM customers WHERE id = %s', (customer_id,))
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def update_customer(self, customer_id, name=None, email=None):
        """Update a customer"""
        if not name and not email:
            return 0
            
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                updates = []
                params = []
                
                if name is not None:
                    updates.append("name = %s")
                    params.append(name)
                if email is not None:
                    updates.append("email = %s")
                    params.append(email)
                
                params.append(customer_id)
                sql = f'UPDATE customers SET {", ".join(updates)} WHERE id = %s'
                cursor.execute(sql, params)
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command}
            ]
            
            response = client.invoke(messages)
            result = response.content.strip()
            print(f"🤖 CustomerAgent Groq response: {result}")

            parsed = json.loads(result)
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
                        'id': c['id'],
                        'name': c['name'],
                        'email': c['email'],
                        'created_at': c['created_at'].isoformat() if c['created_at'] else None
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
                    # Convert datetime to string for JSON serialization
                    if customer.get('created_at'):
                        customer['created_at'] = customer['created_at'].isoformat()
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

    @skill(
        name="manage_customers",
        description="Add, update, list, and delete customer records with PostgreSQL",
        examples=[
            "Add John Doe to customers",
            "Add customer with name Sarah Smith and email sarah@example.com",
            "List all customers",
            "Get customer 1",
            "Update customer 2 name to Michael Johnson",
            "Update customer 2 email to mike@example.com",
            "Delete customer 3"
        ]
    )
    def manage_customers_skill(self, command: str) -> dict:
        return self.process_customer_command(command)

    def ask(self, message):
        """Handle A2A ask requests - CRITICAL for inter-agent communication"""
        print(f"📞 CustomerAgent received ask request: {message}")
        result = self.process_customer_command(message)
        print(f"📤 CustomerAgent sending response: {result}")
        return result

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


if __name__ == '__main__':
    # Get port from environment variable for cloud deployment
    port = int(os.environ.get("PORT", 5002))
    print(f"🚀 Starting CustomerAgent with Groq AI and PostgreSQL on port {port}...")
    agent = CustomerAgent()
    print(f"👤 CustomerAgent running on port {port}")
    run_server(agent, host='0.0.0.0', port=port)
