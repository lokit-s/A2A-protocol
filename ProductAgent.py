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
    name="ProductAgent",
    description="Manages product database operations with pricing using natural language and PostgreSQL",
    version="4.0.0"
)
class ProductAgent(A2AServer):
    def __init__(self):
        super().__init__()
        self.database_url = os.environ.get("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        self.init_database()
        print("📦 ProductAgent database initialized with PostgreSQL and pricing")

    def get_connection(self):
        """Get a new database connection"""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def init_database(self):
        """Initialize the database schema"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS products (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        price DECIMAL(10,2) NOT NULL DEFAULT 0.00,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                print("✅ Products table created/verified with pricing")
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def add_product(self, name, price, description=None):
        """Add a product with price"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    'INSERT INTO products (name, description, price) VALUES (%s, %s, %s) RETURNING id',
                    (name, description, float(price))
                )
                product_id = cursor.fetchone()['id']
                conn.commit()
                return product_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def list_products(self):
        """List all products with pricing"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, name, description, price, created_at FROM products ORDER BY id')
                return cursor.fetchall()
        finally:
            conn.close()

    def get_product(self, product_id):
        """Get a specific product with pricing"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('SELECT id, name, description, price, created_at FROM products WHERE id = %s', (product_id,))
                row = cursor.fetchone()
                if row:
                    result = dict(row)
                    result['price'] = float(result['price'])
                    return result
                return None
        finally:
            conn.close()

    def delete_product(self, product_id):
        """Delete a product"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute('DELETE FROM products WHERE id = %s', (product_id,))
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def update_product(self, product_id, name=None, description=None, price=None):
        """Update a product with optional price update"""
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
            params.append(float(price))
            
        if not updates:
            return 0
            
        conn = self.get_connection()
        try:
            with conn.cursor() as cursor:
                params.append(product_id)
                sql = f'UPDATE products SET {", ".join(updates)} WHERE id = %s'
                cursor.execute(sql, params)
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def process_product_command(self, command: str) -> dict:
        """Process natural language commands for product management with pricing"""
        print(f"🔍 ProductAgent received command: {command}")

        system_prompt = """
You are an assistant that converts user requests about products into structured JSON commands.
Products now have pricing information.
Supported commands:
- To add a product: {"intent":"add_product","parameters":{"name":"product name","price":99.99,"description":"optional description"}}
- To list products: {"intent":"list_products","parameters":{}}
- To get a product: {"intent":"get_product","parameters":{"id": product_id}}
- To delete a product: {"intent":"delete_product","parameters":{"id": product_id}}
- To update a product: {"intent":"update_product","parameters":{"id": product_id, "name": "new name (optional)", "price": 149.99, "description": "new description (optional)"}}

Examples:
- "Add iPhone for $999" -> {"intent":"add_product","parameters":{"name":"iPhone","price":999.00}}
- "Add MacBook Pro for $1299 with description High-performance laptop" -> {"intent":"add_product","parameters":{"name":"MacBook Pro","price":1299.00,"description":"High-performance laptop"}}
- "Update product 1 price to $899" -> {"intent":"update_product","parameters":{"id":1,"price":899.00}}

Return only the JSON, no extra text.
"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": command}
            ]
            
            response = client.invoke(messages)
            result = response.content.strip()
            print(f"🤖 ProductAgent Groq response: {result}")

            parsed = json.loads(result)
            intent = parsed.get("intent")
            params = parsed.get("parameters", {})

            if intent == "add_product":
                name = params.get("name", "").strip()
                price = params.get("price")
                if not name:
                    raise ValueError("Product name missing")
                if price is None:
                    raise ValueError("Product price missing")
                try:
                    price = float(price)
                    if price < 0:
                        raise ValueError("Price cannot be negative")
                except (ValueError, TypeError):
                    raise ValueError("Invalid price format")
                
                description = params.get("description", None)
                product_id = self.add_product(name, price, description)
                return {
                    'status': 'success',
                    'action': 'add_product',
                    'message': f'Product "{name}" added with price ${price:.2f}',
                    'product': {
                        'id': product_id, 
                        'name': name, 
                        'description': description,
                        'price': price
                    }
                }

            elif intent == "list_products":
                products = self.list_products()
                formatted_products = [
                    {
                        'id': p['id'],
                        'name': p['name'],
                        'description': p['description'],
                        'price': float(p['price']),
                        'created_at': p['created_at'].isoformat() if p['created_at'] else None
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
                    # Convert datetime to string for JSON serialization
                    if product.get('created_at'):
                        product['created_at'] = product['created_at'].isoformat()
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
                
                if price is not None:
                    try:
                        price = float(price)
                        if price < 0:
                            raise ValueError("Price cannot be negative")
                    except (ValueError, TypeError):
                        raise ValueError("Invalid price format")
                
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

    @skill(
        name="manage_products",
        description="Add, update, list, and delete product records with pricing using PostgreSQL",
        examples=[
            "Add iPhone for $999",
            "Add MacBook Pro for $1299 with description High-performance laptop",
            "List all products",
            "Get product 1",
            "Update product 2 name to Samsung Galaxy",
            "Update product 2 price to $899",
            "Delete product 3"
        ]
    )
    def manage_products_skill(self, command: str) -> dict:
        return self.process_product_command(command)

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


if __name__ == '__main__':
    # Get port from environment variable for cloud deployment
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Starting ProductAgent with Groq AI, PostgreSQL and Pricing on port {port}...")
    agent = ProductAgent()
    print(f"📦 ProductAgent running on port {port}")
    run_server(agent, host='0.0.0.0', port=port)
