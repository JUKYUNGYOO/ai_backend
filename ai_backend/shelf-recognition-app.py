from flask import Flask
from flask_cors import CORS
from flask_compress import Compress
from flask_restx import Api  # Api를 임포트합니다.
from app.inference import Inference
from orion import TwoStageOrionDetector, OrionLogger, ErrorFiles, OrionDataManager
from glob import glob  # 이 부분을 추가


app = Flask(__name__)

CORS(app)
compress = Compress(app)


api = Api(app)

# 'inference.py'에서 정의된 Inference Namespace를 Api에 추가
# 경로 /inference
api.add_namespace(Inference, '/inference')

@app.route('/health')
def health():
    return {"model_name": f"{model_path}"}
    

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)