import json
import requests

# Load JSON data (replace 'products.json' with your file path)
with open('products.json', 'r') as file:
    data = json.load(file)

# List to store results
results = []

print("Checking URLs...\n")

# Loop through each product and check URL status
for product in data["products"]:
    url = product["url"]
    try:
        response = requests.head(url, timeout=5)  # lightweight HEAD request
        status = response.status_code
        if status == 200:
            result = f"✅ ACTIVE - {url}"
        else:
            result = f"⚠️ INACTIVE ({status}) - {url}"
    except requests.RequestException as e:
        result = f"❌ ERROR - {url} ({e})"

    results.append(result)
    print(result)

# Optionally, save the results to a text file
with open("url_status_report.txt", "w") as report:
    for line in results:
        report.write(line + "\n")

print("\nURL check completed. Results saved to 'url_status_report.txt'.")
