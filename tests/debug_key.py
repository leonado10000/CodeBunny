import jwt
import time
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

def test_key_fix():
    print("--- 1. Generating a valid dummy Private Key ---")
    # Generate a real RSA key so we aren't guessing about format
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    valid_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')
    print("Valid key generated.\n")

    print("--- 2. Simulating Render Environment (The Bug) ---")
    # Render often flattens the key into a single line with literal '\n' characters
    # This is exactly what causes the "Could not parse" error
    mangled_key_from_render = valid_pem.replace('\n', '\\n')
    print(f"Simulated Render Key (First 50 chars): {mangled_key_from_render[:50]}...")
    print("This represents what os.environ['PR_AGENT_PRIVATE_KEY'] looks like on the server.\n")

    print("--- 3. Attempting Encoding WITHOUT Fix ---")
    payload = {'iat': int(time.time()), 'exp': int(time.time()) + 600, 'iss': '12345'}
    try:
        # This mirrors your old code
        jwt.encode(payload, mangled_key_from_render, algorithm='RS256')
        print("SUCCESS? (Unexpected)\n")
    except Exception as e:
        print(f"❌ FAILED (As Expected): {e}")
        print("This confirms the error matches your logs.\n")

    print("--- 4. Attempting Encoding WITH Fix ---")
    try:
        # This mirrors the new code in src/auth.py
        # FIX: We find the literal '\n' characters and turn them back into real newlines
        fixed_key = mangled_key_from_render.replace('\\n', '\n').encode()
        
        jwt.encode(payload, fixed_key, algorithm='RS256')
        print("✅ SUCCESS! The fix works.")
        print("The key was successfully reconstructed and used to sign a token.")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_key_fix()