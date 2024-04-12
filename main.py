from flask import Flask, jsonify, request
from src.helper.train_model import predict

app = Flask(__name__)

@app.route('/questions-type', methods=['POST'])
def question_type():
    input_text = request.json['question']
    predicted_label = predict(input_text)[0]
    return jsonify({'question_type': predicted_label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5005)
