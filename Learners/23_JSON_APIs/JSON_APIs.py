# JSON_APIs.py
from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample in-memory data store
users = {
    1: {"name": "Alice", "age": 30},
    2: {"name": "Bob", "age": 25}
}

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify(users)

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = users.get(user_id)
    if user:
        return jsonify({user_id: user})
    else:
        return jsonify({"error": "User not found"}), 404

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'name' not in data or 'age' not in data:
        return jsonify({"error": "Invalid data"}), 400
    new_id = max(users.keys()) + 1
    users[new_id] = {"name": data['name'], "age": data['age']}
    return jsonify({"id": new_id, "user": users[new_id]}), 201

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.get_json()
    user = users.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    user.update(data)
    return jsonify({user_id: user})

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id in users:
        del users[user_id]
        return jsonify({"message": "User deleted"})
    else:
        return jsonify({"error": "User not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
