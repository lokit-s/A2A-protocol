import json
from datetime import datetime
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_groq import ChatGroq
from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState, A2AClient

# Initialize Groq client
client = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
)


@agent(
    name="SalesAgent",
    description="Handles sales transactions with pricing calculations via RouterAgent using PostgreSQL",
    version="5.0.0"
)
class SalesAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        self.init_database()
        print("💰 SalesAgent database initialized with PostgreSQL and pricing")

        # Initialize client for RouterAgent using environment variable
        router_url = os.environ.get("ROUTER_AGENT_URL", "http://localhost:5000")
        self.router_client = A2AClient(router_url)
        print(f"🔗 SalesAgent connected to RouterAgent at {router_url}")

    def get_connection(self):
        """Get a new database connection"""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def init_database(self):
        """Initialize the database schema"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS sales (
                        id SERIAL PRIMARY KEY,
                        customer_id INTEGER NOT NULL,
                        customer_name VARCHAR(255) NOT NULL,
                        product_id INTEGER NOT NULL,
                        product_name VARCHAR(255) NOT NULL,
                        quantity INTEGER NOT NULL,
                        price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
                        total_price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
                        sale_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                print("✅ Sales table created/verified with pricing")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            conn.rollback()
        finally:
            conn.close()

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
                error_msg = router_response.get('message', 'Unknown error') if isinstance(router_response, dict) else str(router_response)
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
                product_price = product_data.get('price', 0.0)
                print(f"✅ SalesAgent: Found product details via router: {product_name} - ${product_price}")
                return product_name, float(product_price)
            else:
                if agent_response:
                    print(f"❌ SalesAgent: Agent error: {agent_response.get('message', 'Unknown error')}")
                return None, None
                
        except Exception as e:
            print(f"❌ SalesAgent: Error getting product details via router: {e}")
            return None, None

    def make_sale(self, customer_id, product_id, quantity):
        """
        Create a new sale record with pricing calculations via RouterAgent
        """
        print(f"🔍 SalesAgent: Starting make_sale with customer_id={customer_id}, product_id={product_id}, quantity={quantity}")

        try:
            # Validate input parameters first
            if not customer_id:
                raise ValueError("Customer ID is required and cannot be None or empty")
            if not product_id:
                raise ValueError("Product ID is required and cannot be None or empty")
            if not quantity or quantity <= 0:
                raise ValueError(f"Quantity must be a positive number, got: {quantity}")

            # Get customer name via RouterAgent with specific error handling
            print(f"📞 SalesAgent: Fetching customer name for ID: {customer_id} via RouterAgent")
            customer_name = self.get_customer_name(customer_id)

            if not customer_name:
                error_msg = f"Customer not found: No customer exists with ID {customer_id}"
                print(f"❌ SalesAgent: {error_msg}")
                raise ValueError(error_msg)

            print(f"✅ SalesAgent: Found customer: {customer_name}")

            # Get product details via RouterAgent with specific error handling
            print(f"📞 SalesAgent: Fetching product details for ID: {product_id} via RouterAgent")
            product_name, product_price = self.get_product_details(product_id)

            if not product_name or product_price is None:
                error_msg = f"Product not found: No product exists with ID {product_id}"
                print(f"❌ SalesAgent: {error_msg}")
                raise ValueError(error_msg)

            print(f"✅ SalesAgent: Found product: {product_name} - ${product_price}")

            # Calculate total price
            total_price = float(product_price) * int(quantity)
            print(f"💰 SalesAgent: Calculated total price: {quantity} × ${product_price} = ${total_price:.2f}")

            # Attempt database insertion with detailed error handling
            print(f"💾 SalesAgent: Inserting sale record into database...")
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        'INSERT INTO sales (customer_id, customer_name, product_id, product_name, quantity, price, total_price) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id',
                        (customer_id, customer_name, product_id, product_name, quantity, float(product_price), float(total_price))
                    )
                    sale_id = cursor.fetchone()['id']
                    conn.commit()

                    print(f"✅ SalesAgent: Sale record created successfully with ID: {sale_id}")
                    return sale_id, customer_name, product_name, float(product_price), float(total_price)

            except Exception as db_error:
                error_msg = f"Database error during sale insertion: {str(db_error)}"
                print(f"❌ SalesAgent: {error_msg}")
                conn.rollback()
                raise ValueError(error_msg)
            finally:
                conn.close()

        except ValueError as ve:
            # Re-raise ValueError with additional context
            detailed_error = f"Sale creation failed - {str(ve)} [customer_id={customer_id}, product_id={product_id}, quantity={quantity}]"
            print(f"❌ SalesAgent: {detailed_error}")
            raise ValueError(detailed_error)

        except Exception as e:
            # Handle any other unexpected errors
            unexpected_error = f"Unexpected error in make_sale: {str(e)} [customer_id={customer_id}, product_id={product_id}, quantity={quantity}]"
            print(f"❌ SalesAgent: {unexpected_error}")
            raise Exception(unexpected_error)

    def list_sales(self):
        """List all sales from database with pricing"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'SELECT id, customer_id, customer_name, product_id, product_name, quantity, price, total_price, sale_time FROM sales ORDER BY id')
                return cursor.fetchall()
        finally:
            conn.close()

    def delete_sale(self, sale_id):
        """Delete a sale record"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM sales WHERE id = %s', (sale_id,))
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def update_sale(self, sale_id, customer_id=None, product_id=None, quantity=None):
        """Update a sale record with new information and recalculate pricing via RouterAgent"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                # Fetch current sale
                cursor.execute('SELECT customer_id, product_id, quantity FROM sales WHERE id = %s', (sale_id,))
                row = cursor.fetchone()
                if not row:
                    return 0

                current_customer_id, current_product_id, current_quantity = row['customer_id'], row['product_id'], row['quantity']

                # Determine new values
                new_customer_id = customer_id if customer_id is not None else current_customer_id
                new_product_id = product_id if product_id is not None else current_product_id
                new_quantity = quantity if quantity is not None else current_quantity

                # Fetch updated names and pricing via RouterAgent if IDs change
                new_customer_name = self.get_customer_name(new_customer_id)
                new_product_name, new_product_price = self.get_product_details(new_product_id)
                
                if not new_customer_name or not new_product_name or new_product_price is None:
                    raise ValueError("Invalid customer or product ID for update")

                # Recalculate total price
                new_total_price = float(new_product_price) * int(new_quantity)

                cursor.execute(
                    '''UPDATE sales
                       SET customer_id   = %s,
                           customer_name = %s,
                           product_id    = %s,
                           product_name  = %s,
                           quantity      = %s,
                           price         = %s,
                           total_price   = %s
                       WHERE id = %s''',
                    (new_customer_id, new_customer_name, new_product_id, new_product_name, 
                     new_quantity, float(new_product_price), float(new_total_price), sale_id)
                )
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def process_sales_command(self, command: str) -> dict:
        """Process natural language commands for sales management with pricing"""
        print(f"🔍 SalesAgent received command: {command}")
        
        system_prompt = """
You are an assistant that converts user requests about sales into structured JSON commands.
Sales now include pricing calculations.
Supported commands:
- Make a sale: {"intent":"make_sale","parameters":{"customer_id":1,"product_id":2,"quantity":20}}
- List all sales: {"intent":"list_sales","parameters":{}}
- Delete a sale: {"intent":"delete_sale","parameters":{"id": sale_id}}
- Update a sale: {"intent":"update_sale","parameters":{"id": sale_id, "customer_id":1, "product_id":2, "quantity":25}}
Return only the JSON, no extra text.
"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command}
            ]
            
            response = client.invoke(messages)
            result = response.content.strip()
            print(f"🤖 SalesAgent Groq response: {result}")
            
            parsed = json.loads(result)
            intent = parsed.get("intent")
            params = parsed.get("parameters", {})

            if intent == "make_sale":
                customer_id = params.get("customer_id")
                product_id = params.get("product_id")
                quantity = params.get("quantity")
                if not (customer_id and product_id and quantity):
                    raise ValueError("Missing required fields for sale")
                
                sale_id, customer_name, product_name, price, total_price = self.make_sale(customer_id, product_id, quantity)
                return {
                    'status': 'success',
                    'action': 'make_sale',
                    'message': f'Sale recorded: {customer_name} (ID {customer_id}) bought {quantity}x {product_name} (ID {product_id}) at ${price:.2f} each = ${total_price:.2f} total',
                    'sale': {
                        'id': sale_id,
                        'customer_id': customer_id,
                        'customer_name': customer_name,
                        'product_id': product_id,
                        'product_name': product_name,
                        'quantity': quantity,
                        'price': price,
                        'total_price': total_price
                    }
                }

            elif intent == "list_sales":
                sales = self.list_sales()
                formatted_sales = [
                    {
                        'id': s['id'],
                        'customer_id': s['customer_id'],
                        'customer_name': s['customer_name'],
                        'product_id': s['product_id'],
                        'product_name': s['product_name'],
                        'quantity': s['quantity'],
                        'price': float(s['price']),
                        'total_price': float(s['total_price']),
                        'sale_time': s['sale_time'].isoformat() if s['sale_time'] else None
                    } for s in sales
                ]
                return {
                    'status': 'success',
                    'action': 'list_sales',
                    'message': f'Found {len(sales)} sale(s)',
                    'sales': formatted_sales,
                    'count': len(sales)
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
                        'message': f'Sale with ID {sale_id} updated with recalculated pricing'
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

    @skill(
        name="manage_sales",
        description="Record, update, list, and delete sales transactions with pricing calculations via RouterAgent using PostgreSQL",
        examples=[
            "Make a sale by customer 1 buys product 2 of quantity 20",
            "customer 1 buys 20 of product 1",
            "Update sale 3 quantity to 25",
            "List all sales",
            "Delete sale 3"
        ]
    )
    def manage_sales_skill(self, command: str) -> dict:
        return self.process_sales_command(command)

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


if __name__ == '__main__':
    # Get port from environment variable for cloud deployment
    port = int(os.environ.get("PORT", 5003))
    print(f"🚀 Starting SalesAgent with RouterAgent integration, Groq AI, PostgreSQL and Pricing on port {port}...")
    agent = SalesAgent()
    print(f"💰 SalesAgent running on port {port}")
    run_server(agent, host='0.0.0.0', port=port)
