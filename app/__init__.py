
# app/__int__.py

from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from flask_mail import Mail
from flask_login import LoginManager

db = SQLAlchemy()
bcrypt = Bcrypt()
jwt = JWTManager()
mail = Mail()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    bcrypt.init_app(app)
    jwt.init_app(app)
    mail.init_app(app)
    login_manager.init_app(app)
    CORS(app)

    # Now import User and register_blueprints AFTER extensions are initialized
    from app.models import User
    from app.home.routes import home_bp
    from app.analysis.routes import analysis_bp
    from app.predictorlstm.routes import predictorlstm_bp
    from app.predictorgru.routes import predictorgru_bp
    from app.predictorrf.routes import predictorrf_bp
    from app.auth.routes import auth_bp

    app.register_blueprint(home_bp, url_prefix='/')
    app.register_blueprint(analysis_bp, url_prefix='/analysis')
    app.register_blueprint(predictorlstm_bp, url_prefix='/predictorlstm') 
    app.register_blueprint(predictorgru_bp, url_prefix='/predictorgru')
    app.register_blueprint(predictorrf_bp, url_prefix='/predictorrf') 
    app.register_blueprint(auth_bp, url_prefix='/auth')

    with app.app_context():
        db.create_all()

    return app

@login_manager.user_loader
def load_user(user_id):
    from app.models import User
    return User.query.get(int(user_id))
