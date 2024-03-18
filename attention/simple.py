from flask import Flask

app = Flask(__name__)

@app.route('/hi')
def hello():
    return 'Welcome to My Watchlist!'


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # ③