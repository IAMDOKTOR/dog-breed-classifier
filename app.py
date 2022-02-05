from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'soon there will be an application'


if __name__ == '__main__':
    app.run()
