import psycopg2
import psycopg2.extras
import json
from datetime import datetime
import os
from openai import OpenAI
from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState, A2AClient

# Initialize OpenAI client with environment variable
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Database configuration from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

@agent(
    name="SalesAgent",
    description="Handles sales transactions using natural language via RouterAgent",
    version="2.2.0"
)
class SalesAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.conn = None
        self.cursor = None
        self.init_database()
        print("💰 SalesAgent PostgreSQL database initialized")

        # Initialize client for RouterAgent with environment variable
        router_url = os.environ.get("ROUTER_AGENT_URL", "http://localhost:5000")
        self.router_client = A2AClient(router_url)
        print(f"🔗 SalesAgent connected to RouterAgent at {router_url}")

    def init_database(self):
        """Initialize PostgreSQL connection and create table if needed"""
        try:
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL environment variable is not set")
            
            self.conn = psycopg2.connect(DATABASE_URL)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Create sales table if it doesn't exist
            self.cursor.execute('''
                                CREATE TABLE IF NOT EXISTS sales
                                (
                                    id
                                    SERIAL
                                    PRIMARY
                                    KEY,
                                    customer_id
                                    INTEGER
                                    NOT
                                    NULL,
                                    customer_name
                                    VARCHAR
                                (
                                    255
                                ) NOT NULL,
                                    product_id INTEGER NOT NULL,
                                    product_name VARCHAR
                                (
                                    255
                                ) NOT NULL,
                                    quantity INTEGER NOT NULL,
                                    price DECIMAL
                                (
                                    10,
                                    2
                                ),
                                    total_cost DECIMAL
                                (
                                    10,
                                    2
                                ),
                                    sale_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                    )
                                ''')

            # Add new columns if they don't exist (for existing tables)
            try:
                self.cursor.execute("ALTER TABLE sales ADD COLUMN IF NOT EXISTS price DECIMAL(10, 2)")
                self.cursor.execute("ALTER TABLE sales ADD COLUMN IF NOT EXISTS total_cost DECIMAL(10, 2)")
            except Exception as alter_error:
                print(f"Note: Price columns might already exist: {alter_error}")

            self.conn.commit()
            print("✅ PostgreSQL sales table ready with price and total_cost columns")
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

    def parse_agent_response(self, router_response):
        """
        Parse the nested response from RouterAgent - handles both string and dict formats
        """
        try:
            print(f"🔍 Parsing router response type: {type(router_response)}")

            # Handle case where router_response is a string (needs to be parsed first)
            if isinstance(router_response, str):
                print("📝 Router response is string, parsing to dict...")
                try:
                    router_response = json.loads(router_response)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse router response string: {e}")
                    return None

            # Now handle the dict format
            if isinstance(router_response, dict) and router_response.get('status') == 'success':
                # The 'response' field contains a JSON string that needs to be parsed
                agent_response_str = router_response.get('response', '{}')
                print(f"📝 Agent response string: {agent_response_str[:100]}...")

                # Parse the JSON string to get the actual agent response
                if isinstance(agent_response_str, str):
                    try:
                        agent_response = json.loads(agent_response_str)
                        print(f"✅ Successfully parsed agent response")
                        return agent_response
                    except json.JSONDecodeError as e:
                        print(f"❌ Failed to parse agent response JSON: {e}")
                        return None
                else:
                    return agent_response_str
            else:
                error_msg = router_response.get('message', 'Unknown error') if isinstance(router_response,
                                                                                          dict) else str(
                    router_response)
                print(f"❌ Router-level error: {error_msg}")
                return None

        except Exception as e:
            print(f"❌ Unexpected error parsing response: {e}")
            print(f"❌ Response type: {type(router_response)}")
            print(f"❌ Response content: {str(router_response)[:200]}...")
            return None

    def get_customer_name(self, customer_id):
        """Get customer name via RouterAgent with robust parsing"""
        try:
            print(f"📞 SalesAgent: Requesting customer {customer_id} via RouterAgent")
            router_response = self.router_client.ask(f"get customer {customer_id}")

            print(f"📤 SalesAgent: Router response type: {type(router_response)}")

            # Parse the nested response properly
            agent_response = self.parse_agent_response(router_response)

            if agent_response and agent_response.get('status') == 'success':
                customer_data = agent_response.get('customer', {})
                customer_name = customer_data.get('name')
                print(f"✅ SalesAgent: Found customer name via router: {customer_name}")
                return customer_name
            else:
                if agent_response:
                    print(f"❌ SalesAgent: Agent error: {agent_response.get('message', 'Unknown error')}")
                return None

        except Exception as e:
            print(f"❌ SalesAgent: Error getting customer name via router: {e}")
            return None

    def get_product_details(self, product_id):
        """Get product name and price via RouterAgent with robust parsing"""
        try:
            print(f"📞 SalesAgent: Requesting product {product_id} via RouterAgent")
            router_response = self.router_client.ask(f"get product {product_id}")

            print(f"📤 SalesAgent: Router response type: {type(router_response)}")

            # Parse the nested response properly
            agent_response = self.parse_agent_response(router_response)

            if agent_response and agent_response.get('status') == 'success':
                product_data = agent_response.get('product', {})
                product_name = product_data.get('name')
                product_price = product_data.get('price')
                print(f"✅ SalesAgent: Found product via router: {product_name} @ ${product_price}")
                return product_name, product_price
            else:
                if agent_response:
                    print(f"❌ SalesAgent: Agent error: {agent_response.get('message', 'Unknown error')}")
                return None, None

        except Exception as e:
            print(f"❌ SalesAgent: Error getting product details via router: {e}")
            return None, None

    def get_product_name(self, product_id):
        """Get product name via RouterAgent (backwards compatibility)"""
        product_name, _ = self.get_product_details(product_id)
        return product_name

    def make_sale(self, customer_id, product_id, quantity):
        """
        Create a new sale record with price calculation via RouterAgent
        """
        print(
            f"🔍 SalesAgent: Starting make_sale with customer_id={customer_id}, product_id={product_id}, quantity={quantity}")

        self.reconnect_if_needed()

        try:
            # Validate input parameters first
            if not customer_id:
                raise ValueError("Customer ID is required and cannot be None or empty")
            if not product_id:
                raise ValueError("Product ID is required and cannot be None or empty")
            if not quantity or quantity <= 0:
                raise ValueError(f"Quantity must be a positive number, got: {quantity}")

            # Get customer name via RouterAgent
            print(f"📞 SalesAgent: Fetching customer name for ID: {customer_id} via RouterAgent")
            customer_name = self.get_customer_name(customer_id)

            if not customer_name:
                error_msg = f"Customer not found: No customer exists with ID {customer_id}"
                print(f"❌ SalesAgent: {error_msg}")
                raise ValueError(error_msg)

            print(f"✅ SalesAgent: Found customer: {customer_name}")

            # Get product details (name and price) via RouterAgent
            print(f"📞 SalesAgent: Fetching product details for ID: {product_id} via RouterAgent")
            product_name, product_price = self.get_product_details(product_id)

            if not product_name:
                error_msg = f"Product not found: No product exists with ID {product_id}"
                print(f"❌ SalesAgent: {error_msg}")
                raise ValueError(error_msg)

            print(f"✅ SalesAgent: Found product: {product_name}")

            # Calculate total cost
            if product_price is not None:
                total_cost = float(product_price) * quantity
                print(f"💰 SalesAgent: Calculated total cost: ${total_cost:.2f} ({quantity} x ${product_price})")
            else:
                total_cost = None
                print(f"⚠️ SalesAgent: No price available for product, total cost will be NULL")

            # Attempt database insertion
            print(f"💾 SalesAgent: Inserting sale record into database...")
            try:
                self.cursor.execute(
                    'INSERT INTO sales (customer_id, customer_name, product_id, product_name, quantity, price, total_cost) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id',
                    (customer_id, customer_name, product_id, product_name, quantity, product_price, total_cost)
                )
                sale_id = self.cursor.fetchone()['id']
                self.conn.commit()

                print(f"✅ SalesAgent: Sale record created successfully with ID: {sale_id}")
                return sale_id, customer_name, product_name, product_price, total_cost

            except Exception as db_error:
                error_msg = f"Database error during sale insertion: {str(db_error)}"
                print(f"❌ SalesAgent: {error_msg}")
                self.conn.rollback()
                raise ValueError(error_msg)

        except ValueError as ve:
            detailed_error = f"Sale creation failed - {str(ve)} [customer_id={customer_id}, product_id={product_id}, quantity={quantity}]"
            print(f"❌ SalesAgent: {detailed_error}")
            raise ValueError(detailed_error)

        except Exception as e:
            unexpected_error = f"Unexpected error in make_sale: {str(e)} [customer_id={customer_id}, product_id={product_id}, quantity={quantity}]"
            print(f"❌ SalesAgent: {unexpected_error}")
            raise Exception(unexpected_error)

    def list_sales(self):
        """List all sales from database with price and total cost"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute(
                'SELECT id, customer_id, customer_name, product_id, product_name, quantity, price, total_cost, sale_time FROM sales ORDER BY id')
            rows = self.cursor.fetchall()
            return [(row['id'], row['customer_id'], row['customer_name'], row['product_id'],
                     row['product_name'], row['quantity'],
                     float(row['price']) if row['price'] else None,
                     float(row['total_cost']) if row['total_cost'] else None,
                     str(row['sale_time'])) for row in rows]
        except Exception as e:
            print(f"❌ List sales error: {e}")
            raise

    def delete_sale(self, sale_id):
        """Delete a sale record"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('DELETE FROM sales WHERE id = %s', (sale_id,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            return deleted_count
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Delete sale error: {e}")
            raise

    def update_sale(self, sale_id, customer_id=None, product_id=None, quantity=None):
        """Update a sale record with new information and recalculate total cost"""
        self.reconnect_if_needed()
        try:
            # Fetch current sale
            self.cursor.execute('SELECT customer_id, product_id, quantity FROM sales WHERE id = %s', (sale_id,))
            row = self.cursor.fetchone()
            if not row:
                return 0

            current_customer_id, current_product_id, current_quantity = row['customer_id'], row['product_id'], row[
                'quantity']

            # Determine new values
            new_customer_id = customer_id if customer_id is not None else current_customer_id
            new_product_id = product_id if product_id is not None else current_product_id
            new_quantity = quantity if quantity is not None else current_quantity

            # Fetch updated details via RouterAgent
            new_customer_name = self.get_customer_name(new_customer_id)
            new_product_name, new_product_price = self.get_product_details(new_product_id)

            if not new_customer_name or not new_product_name:
                raise ValueError("Invalid customer or product ID for update")

            # Calculate new total cost
            if new_product_price is not None:
                new_total_cost = float(new_product_price) * new_quantity
            else:
                new_total_cost = None

            self.cursor.execute(
                '''UPDATE sales
                   SET customer_id   = %s,
                       customer_name = %s,
                       product_id    = %s,
                       product_name  = %s,
                       quantity      = %s,
                       price         = %s,
                       total_cost    = %s
                   WHERE id = %s''',
                (new_customer_id, new_customer_name, new_product_id, new_product_name, new_quantity, new_product_price,
                 new_total_cost, sale_id)
            )
            updated_count = self.cursor.rowcount
            self.conn.commit()
            return updated_count
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Update sale error: {e}")
            raise

    @skill(
        name="manage_sales",
        description="Record, update, list, and delete sales transactions via RouterAgent",
        examples=[
            "Make a sale by customer 1 buys product 2 of quantity 20",
            "Update sale 3 quantity to 25",
            "List all sales",
            "Delete sale 3"
        ]
    )
    def process_sales_command(self, command: str) -> dict:
        """Process natural language commands for sales management"""
        print(f"🔍 SalesAgent received command: {command}")

        system_prompt = """
You are an assistant that converts user requests about sales into structured JSON commands.
Supported commands:
- Make a sale: {"intent":"make_sale","parameters":{"customer_id":1,"product_id":2,"quantity":20}}
- List all sales: {"intent":"list_sales","parameters":{}}
- Delete a sale: {"intent":"delete_sale","parameters":{"id": sale_id}}
- Update a sale: {"intent":"update_sale","parameters":{"id": sale_id, "customer_id":1, "product_id":2, "quantity":25}}
Return only the JSON, no extra text.
"""
        try:
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.1")),
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "200"))
            )
            result = response.choices[0].message.content.strip()
            print(f"🤖 SalesAgent LLM response: {result}")

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

            if intent == "make_sale":
                customer_id = params.get("customer_id")
                product_id = params.get("product_id")
                quantity = params.get("quantity")
                if not (customer_id and product_id and quantity):
                    raise ValueError("Missing required fields for sale")

                sale_id, customer_name, product_name, product_price, total_cost = self.make_sale(customer_id,
                                                                                                 product_id, quantity)

                # Format message with price information
                if total_cost is not None:
                    message = f'Sale recorded: {customer_name} bought {product_name} (qty: {quantity}) @ ${product_price} each = ${total_cost:.2f} total'
                else:
                    message = f'Sale recorded: {customer_name} bought {product_name} (qty: {quantity}) - price not available'

                return {
                    'status': 'success',
                    'action': 'make_sale',
                    'message': message,
                    'sale': {
                        'id': sale_id,
                        'customer_id': customer_id,
                        'customer_name': customer_name,
                        'product_id': product_id,
                        'product_name': product_name,
                        'quantity': quantity,
                        'price': product_price,
                        'total_cost': total_cost
                    }
                }

            elif intent == "list_sales":
                sales = self.list_sales()
                formatted_sales = [
                    {
                        'id': s[0],
                        'customer_id': s[1],
                        'customer_name': s[2],
                        'product_id': s[3],
                        'product_name': s[4],
                        'quantity': s[5],
                        'price': s[6],
                        'total_cost': s[7],
                        'sale_time': s[8]
                    } for s in sales
                ]

                # Calculate grand total
                grand_total = sum(s[7] for s in sales if s[7] is not None)

                return {
                    'status': 'success',
                    'action': 'list_sales',
                    'message': f'Found {len(sales)} sale(s) with grand total: ${grand_total:.2f}',
                    'sales': formatted_sales,
                    'count': len(sales),
                    'grand_total': grand_total
                }

            elif intent == "delete_sale":
                sale_id = params.get("id")
                if not sale_id:
                    raise ValueError("Sale ID missing")
                deleted = self.delete_sale(sale_id)
                if deleted:
                    return {
                        'status': 'success',
                        'action': 'delete_sale',
                        'message': f'Sale with ID {sale_id} deleted'
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'delete_sale',
                        'message': f'No sale found with ID {sale_id}'
                    }

            elif intent == "update_sale":
                sale_id = params.get("id")
                if not sale_id:
                    raise ValueError("Sale ID missing")
                customer_id = params.get("customer_id")
                product_id = params.get("product_id")
                quantity = params.get("quantity")
                updated = self.update_sale(sale_id, customer_id, product_id, quantity)
                if updated:
                    return {
                        'status': 'success',
                        'action': 'update_sale',
                        'message': f'Sale with ID {sale_id} updated with recalculated total cost'
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'update_sale',
                        'message': f'No sale found with ID {sale_id} or nothing to update'
                    }

            else:
                return {
                    'status': 'error',
                    'action': 'unknown',
                    'message': 'Command not recognized'
                }

        except Exception as e:
            print(f"❌ SalesAgent error: {e}")
            return {
                'status': 'error',
                'action': 'parse_command',
                'message': f'Command failed: {str(e)}'
            }

    def ask(self, message):
        """Handle A2A ask requests - CRITICAL for inter-agent communication"""
        print(f"📞 SalesAgent received ask request: {message}")
        result = self.process_sales_command(message)
        print(f"📤 SalesAgent sending response: {result}")
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
                                                      "text": "Please provide a sales management command."}}
            )
            return task

        result = self.process_sales_command(text)

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
    port = int(os.environ.get('PORT', 5003))
    host = os.environ.get('HOST', '0.0.0.0')  # Render requires 0.0.0.0
    
    print("🚀 Starting SalesAgent with RouterAgent integration...")
    agent = SalesAgent()
    print(f"💰 SalesAgent running on {host}:{port}")
    run_server(agent, host=host, port=port)
