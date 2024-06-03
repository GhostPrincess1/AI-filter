from flask import Flask, send_file
import os

app = Flask(__name__)

@app.route('/get_image', methods=['GET'])
def get_image():
    # 图片文件的路径
    image_path = r'F:\Create-AR-filters-using-Mediapipe\test\9461.jpg_wh300.jpg'  # 替换为您的图片路径

    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
