from flask import Flask,request,jsonify
from flask_cors import CORS
import movie_recommender

app = Flask(__name__)
CORS(app)

@app.route('/movie', methods=['GET'])
def get_recommendations():
    res = movie_recommender.results(request.args.get('title'))
    return jsonify(res)


@app.route('/')
def homepage():
    return(
        f"Welcome to the Movie Recommender API!</br>"
        f"Enter a Movie that You Like After the '=' in the url and the Top 10 Recommendations Will Populate</br>"
        f"Starting Route:</br>"
        f"/movie?title="
    )



if __name__=='__main__':
    app.run(port= 5000, debug= True)