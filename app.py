from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("best_model.pkl")

# Try to detect how many features the model expects
try:
    N_FEATURES = int(getattr(model, "n_features_in_", 3))
except Exception:
    N_FEATURES = 3


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        views = float(request.form["views"])
        likes = float(request.form["likes"])
        comments = float(request.form["comments"])

        # Extra engineered features (if the model needs more than 3)
        like_ratio = likes / views if views > 0 else 0
        comment_ratio = comments / views if views > 0 else 0
        engagement = like_ratio + comment_ratio

        base_features = [views, likes, comments]
        extra_features = [like_ratio, comment_ratio, engagement]

        # Combine and adapt to model's expected feature count
        full_features = base_features + extra_features

        if N_FEATURES <= len(full_features):
            final_features = full_features[:N_FEATURES]
        else:
            # Pad with zeros if model expects more features
            final_features = full_features + [0.0] * (N_FEATURES - len(full_features))

        input_data = np.array([final_features])

        # Predict
        prediction = model.predict(input_data)[0]

        if int(prediction) == 1:
            result = "🚀 This video is LIKELY to TREND in the United States!"
        else:
            result = "❌ This video is NOT very likely to trend in the United States."

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)