from flask import Flask

app = Flask(__name__)


@app.route('/test', methods=["POST"])
def hello_world():
    return {"success": True}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
