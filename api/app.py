from flask import Flask, jsonify
import requests
import imgCompress
app=Flask(__name__)

@app.route('/get_image',methods=['GET'])
def get_image():
    encoded_image, original, compressed, ratio = imgCompress.return_base64Image()

    # Return all variables as a JSON response
    response_data = {
        'encoded_image': encoded_image,
        'original_size': original,
        'compressed_size': compressed,
        'ratio': ratio
    }

    return jsonify(response_data)

    

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5004,debug=True)