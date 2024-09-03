from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/notify_global_weights', methods=['POST'])
def notify_global_weights():
    print("Received notification that global weights are available.")
    # Trigger the download of global weights
    fedclient.download_global_weights()
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)  # Each client should run on a different port
