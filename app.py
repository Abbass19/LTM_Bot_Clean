from ariadne import graphql_sync, make_executable_schema, gql, load_schema_from_path, ObjectType, ScalarType
from flask import request,jsonify,Flask
import json

from api import settings
from api.queries import resolve_trainLTM

json_scalar = ScalarType('JSON')
@json_scalar.serializer
def serialize_json(value):
    return value
@json_scalar.value_parser
def parse_json_value(value):
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
@json_scalar.literal_parser
def parse_json_literal(ast):
    # Assuming AST is a string representing a JSON object
    try:
        return json.loads(ast.value)
    except (json.JSONDecodeError, TypeError):
        return None


app = Flask(__name__)
query = ObjectType("Query")
query.set_field("trainLTM",resolve_trainLTM)

type_defs = load_schema_from_path(settings.GRAPHQL_SCHEMA)
schema = make_executable_schema(
    type_defs,query
)

# GraphQL endpoint
@app.route('/graphql', methods=["POST"])
def graphql():
    data = request.get_json()
    success, result = graphql_sync(schema, data)
    status_code = 200 if success else 400 
    return jsonify(result), status_code


if __name__ == '__main__':
    app.run(host=settings.SERVER_HOST,port=settings.SERVER_PORT,debug=True)