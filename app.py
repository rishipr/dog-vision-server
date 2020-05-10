from flask import Flask, request, jsonify
from flask_cors import CORS
import fastai.vision as fastai
import os


app = Flask(__name__)
CORS(app)

CLASSIFIER = fastai.load_learner('./models', "classifier.pkl")


@app.after_request
def add_headers(response):
    response.headers.add('Content-Type', 'application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods',
                         'PUT, GET, POST, DELETE, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Expose-Headers',
                         'Content-Type,Content-Length,Authorization,X-Pagination')

    return response


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    print(f"Request: {request}")
    files = request.files
    print(f"Files: {files}")
    image = fastai.image.open_image(files['image'])
    print(f"Image: {image}")
    # image = fastai.image.open_image(".test-images/corgi.jpg")

    print(f"Predicting...")
    prediction = CLASSIFIER.predict(image)
    print(f"Prediction done: {prediction}")
    categories = CLASSIFIER.data.classes
    print(f"Categories: {categories}")

    # Get list of predictions
    predictions = list(
        zip(
            categories,
            [round(x, 4) for x in map(float, prediction[2])]
        )
    )

    print(f"predictions: {predictions}")

    # Sort predictions
    predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Return JSON object of prediction results
    response = jsonify({
        "breedPredictions": predictions_sorted
    })

    print(f"response: {response}")

    # response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route("/", methods=["GET", "OPTIONS"])
def hello_world():
    response = jsonify({
        "success": "true"
    })

    # response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    # app.debug = False
    port = int(os.environ.get('PORT', 8000))
    app.run(host="0.0.0.0", port=port)
