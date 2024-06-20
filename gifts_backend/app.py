from flask import Flask, request, jsonify
from flask_cors import CORS
from RecsRepo import RecsRepo

app = Flask(__name__)
CORS(app)
app.config['WTF_CSRF_ENABLED'] = False


@app.route('/get_recs', methods=['POST'])
def get_recommendations_api():
    input_preferences = request.json
    recs_repo = RecsRepo()
    return jsonify(recs_repo.get_recs(input_preferences))


if __name__ == '__main__':
    app.run(debug=True)
