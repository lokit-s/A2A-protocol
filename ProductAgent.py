import psycopg2
import psycopg2.extras
import json
from datetime import datetime
import os

from python_a2a import A2AServer, skill, agent, run_server, TaskStatus, TaskState

client = ChatGroq(
    groq_api_key=os.environ.get("GROQ_API_KEY"),
    model_name=os.environ.get("GROQ_MODEL", "llama3-70b-8192")
)
# Database configuration from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

@agent(
    name="ProductAgent",
    description="Manages product database operations using natural language",
    version="1.2.0"
)
class ProductAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.conn = None
        self.cursor = None
        self.init_database()
        print("📦 ProductAgent PostgreSQL database initialized")

    def init_database(self):
        """Initialize PostgreSQL connection and create table if needed"""
        try:
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL environment variable is not set")
            
            self.conn = psycopg2.connect(DATABASE_URL)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Create products table if it doesn't exist
            self.cursor.execute('''
                                CREATE TABLE IF NOT EXISTS products
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
                                    description TEXT,
                                    price DECIMAL
                                (
                                    10,
                                    2
                                ),
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                    )
                                ''')

            # Add price column if it doesn't exist (for existing tables)
            try:
                self.cursor.execute("ALTER TABLE products ADD COLUMN IF NOT EXISTS price DECIMAL(10, 2)")
            except Exception as alter_error:
                print(f"Note: Price column might already exist: {alter_error}")

            self.conn.commit()
            print("✅ PostgreSQL products table ready with price column")
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

    def add_product(self, name, description=None, price=None):
        """Add a new product to the database"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute(
                'INSERT INTO products (name, description, price) VALUES (%s, %s, %s) RETURNING id',
                (name, description, price)
            )
            product_id = self.cursor.fetchone()['id']
            self.conn.commit()
            return product_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Add product error: {e}")
            raise

    def list_products(self):
        """List all products from the database"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('SELECT id, name, description, price, created_at FROM products ORDER BY id')
            rows = self.cursor.fetchall()
            return [(row['id'], row['name'], row['description'], float(row['price']) if row['price'] else None,
                     str(row['created_at'])) for row in rows]
        except Exception as e:
            print(f"❌ List products error: {e}")
            raise

    def get_product(self, product_id):
        """Get a specific product by ID"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('SELECT id, name, description, price, created_at FROM products WHERE id = %s',
                                (product_id,))
            row = self.cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'price': float(row['price']) if row['price'] else None,
                    'created_at': str(row['created_at'])
                }
            return None
        except Exception as e:
            print(f"❌ Get product error: {e}")
            raise

    def delete_product(self, product_id):
        """Delete a product by ID"""
        self.reconnect_if_needed()
        try:
            self.cursor.execute('DELETE FROM products WHERE id = %s', (product_id,))
            deleted_count = self.cursor.rowcount
            self.conn.commit()
            return deleted_count
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Delete product error: {e}")
            raise

    def update_product(self, product_id, name=None, description=None, price=None):
        """Update a product's information"""
        self.reconnect_if_needed()
        try:
            updates = []
            params = []

            if name is not None:
                updates.append("name = %s")
                params.append(name)
            if description is not None:
                updates.append("description = %s")
                params.append(description)
            if price is not None:
                updates.append("price = %s")
                params.append(price)

            if not updates:
                return 0

            params.append(product_id)
            sql = f'UPDATE products SET {", ".join(updates)} WHERE id = %s'

            self.cursor.execute(sql, params)
            updated_count = self.cursor.rowcount
            self.conn.commit()
            return updated_count
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Update product error: {e}")
            raise

    @skill(
        name="manage_products",
        description="Add, update, list, and delete product records via RouterAgent",
        examples=[
            "Add iPhone to products",
            "Add product MacBook Pro with description High-performance laptop and price 1299.99",
            "Add Samsung Galaxy with price 899.50",
            "List all products",
            "Get product 1",
            "Update product 2 name to Samsung Galaxy and price to 799.99",
            "Delete product 3"
        ]
    )
    def process_product_command(self, command: str) -> dict:
        """Process natural language commands for product management"""
        print(f"🔍 ProductAgent received command: {command}")

        system_prompt = """
You are an assistant that converts user requests about products into structured JSON commands.
Supported commands:
- To add a product: {"intent":"add_product","parameters":{"name":"product name","description":"optional description","price": optional_price_number}}
- To list products: {"intent":"list_products","parameters":{}}
- To get a product: {"intent":"get_product","parameters":{"id": product_id}}
- To delete a product: {"intent":"delete_product","parameters":{"id": product_id}}
- To update a product: {"intent":"update_product","parameters":{"id": product_id, "name": "new name (optional)", "description": "new description (optional)", "price": new_price_number_optional}}

Important: 
- Price should be a number (e.g., 299.99, 1500, 49.95)
- Extract price from phrases like "with price 299", "costs 150", "priced at 99.99"
- If no price is mentioned, don't include price in parameters

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
            print(f"🤖 ProductAgent LLM response: {result}")

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

            if intent == "add_product":
                name = params.get("name", "").strip()
                if not name:
                    raise ValueError("Product name missing")
                description = params.get("description", None)
                price = params.get("price", None)

                # Convert price to float if provided
                if price is not None:
                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        price = None

                product_id = self.add_product(name, description, price)
                return {
                    'status': 'success',
                    'action': 'add_product',
                    'message': f'Product "{name}" added with price ${price}' if price else f'Product "{name}" added',
                    'product': {'id': product_id, 'name': name, 'description': description, 'price': price}
                }

            elif intent == "list_products":
                products = self.list_products()
                formatted_products = [
                    {
                        'id': p[0],
                        'name': p[1],
                        'description': p[2],
                        'price': p[3],
                        'created_at': p[4]
                    } for p in products
                ]
                return {
                    'status': 'success',
                    'action': 'list_products',
                    'message': f'Found {len(products)} product(s)',
                    'products': formatted_products,
                    'count': len(products)
                }

            elif intent == "get_product":
                prod_id = params.get("id")
                if not prod_id:
                    raise ValueError("Product ID missing")
                product = self.get_product(prod_id)
                print(f"🔍 ProductAgent found product: {product}")
                if product:
                    return {
                        'status': 'success',
                        'action': 'get_product',
                        'message': f'Product {prod_id} found',
                        'product': product
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'get_product',
                        'message': f'No product found with ID {prod_id}'
                    }

            elif intent == "delete_product":
                prod_id = params.get("id")
                if not prod_id:
                    raise ValueError("Product ID missing")
                deleted = self.delete_product(prod_id)
                if deleted:
                    return {
                        'status': 'success',
                        'action': 'delete_product',
                        'message': f'Product with ID {prod_id} deleted'
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'delete_product',
                        'message': f'No product found with ID {prod_id}'
                    }

            elif intent == "update_product":
                prod_id = params.get("id")
                if not prod_id:
                    raise ValueError("Product ID missing")
                name = params.get("name", None)
                description = params.get("description", None)
                price = params.get("price", None)

                # Convert price to float if provided
                if price is not None:
                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        price = None

                updated = self.update_product(prod_id, name, description, price)
                if updated:
                    return {
                        'status': 'success',
                        'action': 'update_product',
                        'message': f'Product with ID {prod_id} updated'
                    }
                else:
                    return {
                        'status': 'error',
                        'action': 'update_product',
                        'message': f'No product found with ID {prod_id} or nothing to update'
                    }

            else:
                return {
                    'status': 'error',
                    'action': 'unknown',
                    'message': 'Command not recognized'
                }

        except Exception as e:
            print(f"❌ ProductAgent error: {e}")
            return {
                'status': 'error',
                'action': 'parse_command',
                'message': f'Command failed: {str(e)}'
            }

    def ask(self, message):
        """Handle A2A ask requests - CRITICAL for inter-agent communication"""
        print(f"📞 ProductAgent received ask request: {message}")
        result = self.process_product_command(message)
        print(f"📤 ProductAgent sending response: {result}")
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
                                                      "text": "Please provide a product management command."}}
            )
            return task

        result = self.process_product_command(text)

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
    port = int(os.environ.get('PORT', 5001))
    host = os.environ.get('HOST', '0.0.0.0')  # Render requires 0.0.0.0
    
    agent = ProductAgent()
    print(f"🚀 Product Agent running on {host}:{port}")
    run_server(agent, host=host, port=port)
