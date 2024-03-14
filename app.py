from ariadne import graphql_sync, make_executable_schema, gql, load_schema_from_path, ObjectType
from flask import request,jsonify,Flask

from api import settings
from api.queries import resolve_fitLTMBot, resolve_predictLTMBot

app = Flask(__name__)

query = ObjectType("Query")
query.set_field("fitLTMBot", resolve_fitLTMBot)
query.set_field("predictLTMBot", resolve_predictLTMBot)

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