import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types

app = Flask(__name__)
# Allow CORS for your Vercel frontend URL
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the Gemini Client
# It automatically looks for an environment variable named GEMINI_API_KEY
GEMINI_CLIENT = genai.Client()

@app.route('/optimize-prescription', range=['POST'])
def optimize_prescription():
    try:
        request_data = request.get_json()
        if not request_data or 'image' not in request_data:
            return jsonify({'error': 'Missing image canvas payload'}), 400

        # Decode the incoming Base64 canvas data URL
        raw_image_data = request_data['image']
        if ',' in raw_image_data:
            raw_image_data = raw_image_data.split(',')[1]
        
        image_bytes = base64.b64decode(raw_image_data)
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Structured prompt instructing Gemini to analyze the handwriting/photo
        system_prompt = (
            "You are an advanced clinical medical AI. Analyze this handwritten or snapped prescription image. "
            "Extract the details precisely and return a clean JSON object with these EXACT keys: "
            "'medication', 'dosage', 'duration', and 'instructions'. Do not add markdown formatting outside JSON."
        )

        # Call Gemini 2.5 Flash (Optimized for fast multimodal tasks)
        response = GEMINI_CLIENT.models.generate_content(
            model='gemini-2.5-flash',
            contents=[pil_image, system_prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        # Return the parsed AI result directly back to the frontend
        return response.text, 200, {'Content-Type': 'application/json'}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({'error': 'Internal AI processing crash', 'details': str(e)}), 500

if __name__ == '__main__':
    # Bind to PORT environment variable assigned by the cloud platform
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)