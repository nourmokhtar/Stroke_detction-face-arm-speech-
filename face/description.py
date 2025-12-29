# description.py
# ----------------------------------------------------
# Generates a safe, non-medical visual explanation
# from handcrafted facial symmetry features.
# Also allows sending general prompts to your hosted LLM.
# ----------------------------------------------------

import httpx
from openai import OpenAI

# -------------------------------
# Hosted LLM setup (TokenFactory)
# -------------------------------
def get_client():
    http_client = httpx.Client(verify=False)  # Disable SSL verification
    client = OpenAI(
        api_key="sk-34ef7724dde84849819a687a17406604",  # 👉 replace with your key
        base_url="https://tokenfactory.esprit.tn/api",
        http_client=http_client
    )
    return client

# -------------------------------
# 1️⃣ Explain facial symmetry features safely
def explain_features(features, severity=None):
    """
    features = {
        "smile_vertical_asymmetry": float,
        "mouth_horizontal_asymmetry": float,
        "eye_horizontal_asymmetry": float,
        "general_facial_symmetry": float
    }
    severity = float, optional severity value
    """

    note = ""
    if severity is not None and severity > 15:
        note = "\n⚠️ Note: The overall facial asymmetry is quite pronounced, so some visual differences may be noticeable."

    prompt = f"""
You will receive numerical values describing facial symmetry.
These values are NOT for medical use. 
Your task is to give a clear, strictly non-medical interpretation of visual balance.
Focus on the severity of asymmetry, differences in pose, or expression, 
and describe how noticeable or subtle each difference might appear visually.

Here are the values:
- Smile vertical asymmetry: {features['smile_vertical_asymmetry']}
- Mouth horizontal asymmetry: {features['mouth_horizontal_asymmetry']}
- Eye horizontal asymmetry: {features['eye_horizontal_asymmetry']}
- General facial symmetry: {features['general_facial_symmetry']}{note}

Instructions:
- Categorize each value as: "minimal/subtle", "moderate/noticeable", or "high/obvious" asymmetry.
- If any value is greater than 15, interpret it as making the asymmetry more visually obvious than the number alone suggests.
- Describe what these asymmetries would look like visually (e.g., slight tilt of mouth, one eye slightly higher, uneven smile).
- At the end, provide a single conclusion summarizing the overall face:
  - "Overall, the face appears normal/balanced"
  - "Overall, the face is slightly asymmetrical/noticeable differences"
  - "Overall, the face shows clear asymmetry/more pronounced differences"
- Do NOT mention medical conditions, disorders, or make any diagnosis.
- Keep the interpretation strictly visual and severity-focused.
"""

    client = get_client()
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

# -------------------------------
# 2️⃣ General prompt function
# -------------------------------
def generate_description(prompt):
    """
    Send any text prompt to the hosted LLM and get the response.
    """
    client = get_client()
    response = client.chat.completions.create(
        model="hosted_vllm/Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": "Tu es un assistant utile et concis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Example features
    features_example = {
        "smile_vertical_asymmetry": 0.05,
        "mouth_horizontal_asymmetry": 0.03,
        "eye_horizontal_asymmetry": 0.01,
        "general_facial_symmetry": 0.04
    }

    print("[Visual Description]")
    print(explain_features(features_example))
