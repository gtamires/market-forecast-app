
import os
from app import create_app

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = create_app()

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000, debug=True)

 