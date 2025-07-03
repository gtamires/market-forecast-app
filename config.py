import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
  SECRET_KEY = 'your_secret_key'
  SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
  SQLALCHEMY_TRACK_MODIFICATIONS = False
  JWT_SECRET_KEY = 'your_jwt_secret_key'
  RATELIMIT_DEFAULT = "200 per day;50 per hour"
  RATELIMIT_STRATEGY = 'fixed-window-elastic-expiry'
  RATELIMIT_STORAGE_URL = 'memory://'
  
  MAIL_SERVER = 'smtp.gmail.com'
  MAIL_PORT = 587
  MAIL_USE_TLS = True
  MAIL_USERNAME = 'africanenergyreformation@gmail.com'
  MAIL_PASSWORD = 'oejy ltmx revg tfeh'
  MAIL_USE_TLS = True
  MAIL_USE_SSL = False
  
  FLASKY_MAIL_SUBJECT_PREFIX = '[SSO App]'
  FLASKY_MAIL_SENDER = 'SSO Admin <sso@example.com>'

  EMAIL_SENDER = 'africanenergyreformation@gmail.com'