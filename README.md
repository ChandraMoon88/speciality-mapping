# Specialty Mapping Application

A medical specialty mapping application that helps identify the appropriate medical specialist based on symptoms and body parts.

## Features

- Symptom to specialist mapping
- SNOMED CT terminology integration
- Emergency rule detection
- Organ to specialty mapping

## Project Structure

```
point/
├── app.py                      # Main Flask application
├── index.html                  # Frontend interface
├── requirements.txt            # Python dependencies
├── medical_keywords_clean.json # Medical keywords dictionary
├── snomed_data/                # SNOMED reference data
│   ├── body_hierarchy.csv
│   ├── emergency_rules.json
│   ├── organ_to_specialty_map.csv
│   ├── symptom_master.csv
│   └── symptom_to_body_map.csv
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ChandraMoon88/speciality-mapping.git
cd point
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://127.0.0.1:5000`

## Model Files

The application uses ML model files for predictions. Due to their size, model files are tracked using Git LFS. If models are missing, they can be downloaded separately or pulled using:

```bash
git lfs pull
```

## Technologies

- **Backend:** Python, Flask
- **ML Models:** ONNX runtime
- **Medical Terminology:** SNOMED CT
- **Frontend:** HTML, CSS, JavaScript

## License

MIT License
