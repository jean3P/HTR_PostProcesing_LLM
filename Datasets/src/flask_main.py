# flask_main.py

import logging
from flask import Flask
from flask_cors import CORS
from graphql_server.flask import GraphQLView
from my_graphql.schema import schema

# Configure logging for the main module
logging.basicConfig(level=logging.INFO)

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for specific route (/graphql) and origin (localhost:4200)
CORS(app, resources={r"/graphql": {"origins": "http://localhost:4200"}})
# Add the GraphQL endpoint
app.add_url_rule(
    '/graphql',
    view_func=GraphQLView.as_view(
        'graphql',
        schema=schema,
        graphiql=True  # Enable GraphiQL interface
    )
)

# Run the Flask app
if __name__ == '__main__':

    # Start the Flask app
    logging.info("Running app")
    app.run(host='0.0.0.0', port=5000, debug=True)
