from ariadne import graphql_sync, make_executable_schema, gql, load_schema_from_path, ObjectType
from flask import request,jsonify,Flask

from api import settings
from api.queries import resolve_fitLongShortTermMemory, resolve_consumeLongShortTermMemory

app = Flask(__name__)
query = ObjectType("Query")
query.set_field("fitLongShortTermMemory", resolve_fitLongShortTermMemory)
query.set_field("consumeLongShortTermMemory", resolve_consumeLongShortTermMemory)

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