from flask import render_template, Blueprint


main = Blueprint('main', __name__)

@main.route('/')
@main.route('/home')
def home():
    return render_template('main/home.html', title='Homepage')


@main.route('/about')
def about():
    return render_template('main/about.html', title='About')
