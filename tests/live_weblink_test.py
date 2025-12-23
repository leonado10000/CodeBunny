import requests
import json

# --- CONFIGURATION ---
# UNCOMMENT THIS FOR LIVE RENDER TESTING
# SERVICE_URL = "https://codebunny-5o9o.onrender.com" 

# UNCOMMENT THIS FOR LOCAL DOCKER TESTING
# NOTE: On Windows, use http://127.0.0.1:8080, NOT http://0.0.0.0:8080
SERVICE_URL = "https://codebunny-5o9o.onrender.com"

ENDPOINT = f"{SERVICE_URL}/webhook/github"

# We need to simulate the data GitHub Actions sends: ${{ toJSON(github) }}
# IMPORTANT: Use a repository name where you have actually INSTALLED the App.
TEST_REPO = "leonado10000/CodeBunny" 
TEST_PR_NUMBER = 1 # Use a real PR number if possible to fetch a real diff

mock_payload = {
    # The 'repository' field comes as a string "owner/repo" in the Actions context
    "repository": TEST_REPO,
    "event": {
        "pull_request": {
            "number": TEST_PR_NUMBER,
            # We provide a dummy diff URL. The code might fail fetching this if it's not real,
            # but it will pass the Authentication step first, which is what we want to test.
            "diff_url": f"https://github.com/{TEST_REPO}/pull/{TEST_PR_NUMBER}.diff",
            "base": {
                "ref": "main"
            }
        },
        # Some event payloads include installation info, though our code fetches it manually
        "installation": {
            "id": 000000
        }
    }
}

def run_live_test():
    print(f"🚀 Sending payload to: {ENDPOINT}")
    print(f"📦 Payload Repo: {TEST_REPO}")
    
    try:
        response = requests.post(ENDPOINT, json=mock_payload)
        
        print("\n--- RESPONSE ---")
        print(f"Status Code: {response.status_code}")
        
        try:
            # Try to print pretty JSON
            print(json.dumps(response.json(), indent=2))
        except:
            # If plain text or HTML error
            print(response.text)
            
        if response.status_code == 200:
            print("\n✅ SUCCESS: The server accepted the payload and processed it.")
        elif response.status_code == 500:
            print("\n❌ SERVER ERROR (500): Check your Docker Terminal Logs.")
            print("This usually means Authentication failed or the Key is still wrong.")
        else:
            print(f"\n⚠️ UNEXPECTED STATUS: {response.status_code}")

    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("   -> Is Docker running? Did you run 'docker run --env-file local.env -p 8080:8080 codebunny-local'?")

if __name__ == "__main__":
    run_live_test()