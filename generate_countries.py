# generate_countries.py
import pycountry
import json

def get_flag_emoji(country_code):
    """Converts a 2-letter country code to a flag emoji."""
    try:
        return ''.join([chr(ord(char) + 127397) for char in country_code.upper()])
    except:
        return ""

countries = []
for country in pycountry.countries:
    country_entry = {
        "name": country.name,
        "code": country.alpha_3,
        "flag": get_flag_emoji(country.alpha_2)
    }
    countries.append(country_entry)

# Handle special cases or missing flags if necessary
special_countries = [
    {"name": "Korea, North", "code": "PRK", "flag": get_flag_emoji("KP")},
    {"name": "Korea, South", "code": "KOR", "flag": get_flag_emoji("KR")},
    {"name": "Czechia (Czech Republic)", "code": "CZE", "flag": get_flag_emoji("CZ")},
    {"name": "Taiwan", "code": "TWN", "flag": get_flag_emoji("TW")},
    {"name": "Vatican City", "code": "VAT", "flag": get_flag_emoji("VA")},
    {"name": "Eswatini (Swaziland)", "code": "SWZ", "flag": get_flag_emoji("SZ")},
    # Add any other special cases as needed
]

# Update or add special countries
for special in special_countries:
    exists = False
    for country in countries:
        if country['code'] == special['code']:
            country['name'] = special['name']  # Update name
            country['flag'] = special['flag']  # Update flag if needed
            exists = True
            break
    if not exists:
        countries.append(special)

# Save to countries.json
with open("countries.json", "w", encoding="utf-8") as f:
    json.dump(countries, f, ensure_ascii=False, indent=4)

print("countries.json has been generated successfully.")
