
from flask import Blueprint, render_template

home_bp = Blueprint('home', __name__)

# This is the home page
@home_bp.route('/')
def index():
    return render_template('home/index.html') 

