
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import onnxruntime as ort
from transformers import AutoTokenizer
import pickle
import json
import os
import tempfile
import uuid
from datetime import datetime
import numpy as np
import re
import logging
import csv
import difflib

import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
import io
import asyncio

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("⚠ edge-tts not available (pip install edge-tts), using pyttsx3 only")

SILERO_AVAILABLE = False


try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    FASTTEXT_AVAILABLE = False
    print("⚠ fasttext not available, language detection disabled")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠ llama-cpp-python not available, multilingual LLM disabled")

try:
    from deep_translator import MyMemoryTranslator, GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠ deep-translator not available, using LLM for translation")

try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✓ Whisper available (100+ languages STT)")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠ whisper not available (pip install openai-whisper)")

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
    print("✓ Vosk available (offline STT)")
except ImportError:
    VOSK_AVAILABLE = False
    print("⚠ vosk not available (pip install vosk)")


from huggingface_hub import hf_hub_download


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "medical model")
CACHE_DIR = os.path.join(BASE_DIR, ".model_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

QWEN_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
QWEN_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"


class LanguageDetector:
    
    def __init__(self, cache_dir: str):
        self.model = None
        self.cache_dir = cache_dir
        self.model_path = os.path.join(cache_dir, "lid.176.bin")
        
        if not FASTTEXT_AVAILABLE:
            logger.warning("fastText not available - language detection disabled")
            return
        
        self._load_model()
    
    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.info("Downloading fastText language detection model...")
            import urllib.request
            urllib.request.urlretrieve(FASTTEXT_MODEL_URL, self.model_path)
            logger.info("✓ Downloaded lid.176.bin")
        
        self.model = fasttext.load_model(self.model_path)
        logger.info("✓ Loaded fastText language detector (176 languages)")
    
    def detect(self, text: str) -> Dict:
        if not self.model:
            return {"language": "en", "confidence": 0.0, "detected": False}
        
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return {"language": "en", "confidence": 0.0, "detected": False}
        
        predictions = self.model.predict(clean_text, k=3)
        
        top_lang = predictions[0][0].replace("__label__", "")
        top_conf = float(predictions[1][0])
        
        return {
            "language": top_lang,
            "confidence": top_conf,
            "detected": True,
            "top_3": [
                {"lang": predictions[0][i].replace("__label__", ""), "conf": float(predictions[1][i])}
                for i in range(min(3, len(predictions[0])))
            ]
        }


class MedicalTranslator:
    
    LANG_MAP = {
        'zh': 'zh-CN',
    }
    
    def __init__(self, cache_dir: str = None):
        self.available = TRANSLATOR_AVAILABLE
        if not self.available:
            logger.warning("Google Translator (deep-translator) not available - using LLM fallback")
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        if source_lang == "en" or not text.strip():
            return text
        
        if not self.available:
            return text
        
        try:
            src = self.LANG_MAP.get(source_lang, source_lang)
            
            try:
                translator = GoogleTranslator(source='auto', target='en')
                translated = translator.translate(text)
                logger.info(f"Google Translate ({source_lang}→en): '{text[:30]}...'")
                return translated
            except Exception as e:
                logger.warning(f"Google Translate failed: {e}. Trying MyMemory...")
                translator = MyMemoryTranslator(source=src, target='en')
                translated = translator.translate(text)
                logger.info(f"MyMemory Translate ({source_lang}→en): Success")
                return translated
        
        except Exception as e:
            logger.error(f"Translation error (All providers): {e}")
            return text
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang == "en" or not text.strip():
            return text
        
        if not self.available:
            return text
        
        try:
            tgt = self.LANG_MAP.get(target_lang, target_lang)
            
            try:
                translator = GoogleTranslator(source='en', target=tgt)
                translated = translator.translate(text)
                logger.info(f"Google Translate (en→{target_lang}): Success")
                return translated
            except Exception as e:
                logger.warning(f"Google Translate failed: {e}. Trying MyMemory...")
                translator = MyMemoryTranslator(source='en', target=tgt)
                translated = translator.translate(text)
                logger.info(f"MyMemory Translate (en→{target_lang}): Success")
                return translated
        
        except Exception as e:
            logger.error(f"Translation error (All providers): {e}")
            return text


class MultilingualBrain:
    
    def __init__(self, cache_dir: str):
        self.model = None
        self.cache_dir = cache_dir
        
        if not LLAMA_CPP_AVAILABLE:
            logger.warning("llama-cpp-python not available - multilingual LLM disabled")
            return
        
        self._load_model()
    
    def _load_model(self):
        model_path = os.path.join(self.cache_dir, QWEN_MODEL_FILE)
        
        if not os.path.exists(model_path):
            logger.info(f"Downloading {QWEN_MODEL_FILE} (~1GB)...")
            model_path = hf_hub_download(
                repo_id=QWEN_MODEL_REPO,
                filename=QWEN_MODEL_FILE,
                cache_dir=self.cache_dir,
                local_dir=self.cache_dir
            )
            logger.info("✓ Downloaded Qwen2.5-1.5B-Instruct")
        
        logger.info("Loading Qwen2.5-1.5B-Instruct (this may take 10-30 seconds)...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )
        logger.info("✓ Loaded Qwen2.5-1.5B-Instruct (multilingual)")
    
    LANG_NAME_MAP = {
        'te': 'Telugu', 'hi': 'Hindi', 'ta': 'Tamil', 'ml': 'Malayalam', 'kn': 'Kannada',
        'bn': 'Bengali', 'gu': 'Gujarati', 'mr': 'Marathi', 'pa': 'Punjabi', 'ur': 'Urdu',
        'es': 'Spanish', 'fr': 'French', 'de': 'German', 'zh': 'Chinese', 'ar': 'Arabic'
    }

    def generate_response(
        self, 
        symptoms: str, 
        language: str = "en",
        specialist: str = "General Medicine",
        secondary_specialists: list = None,
        severity: str = "moderate",
        max_tokens: int = 600
    ) -> str:
        
        target_lang = language
        
        if not self.model:
            return self._fallback_response(symptoms, target_lang, specialist, severity, secondary_specialists)
            
        
        specs = [specialist]
        if secondary_specialists:
            for s in secondary_specialists:
                if s not in specs: specs.append(s)
        
        fallbacks = ["General Medicine", "Internal Medicine", "Family Physician"]
        for fb in fallbacks:
            if len(specs) >= 3: break
            if fb not in specs: specs.append(fb)
        
        specs = specs[:3]
        
        system_prompt = f"""You are a medical triage assistant. You MUST provide exactly 3 specialists.

CRITICAL INSTRUCTIONS:
- PRIORITY 1 MUST BE: {specs[0]}
- PRIORITY 2 MUST BE: {specs[1]}
- PRIORITY 3 MUST BE: {specs[2]}
- You are FORBIDDEN from choosing different specialists.

REQUIRED FORMAT:

I understand you're experiencing [brief symptom acknowledgment].

🥇 Priority 1: {specs[0]} - Score: [Score 0-100]%
Reasoning: [Write EXACTLY 1 single sentence explaining why (max 20 words).]

🥈 Priority 2: {specs[1]} - Score: [Score 0-100]%
Reasoning: [Write EXACTLY 1 single sentence (max 15 words).]

🥉 Priority 3: {specs[2]} - Score: [Score 0-100]%
Reasoning: [Write EXACTLY 1 single sentence (max 15 words).]

Remember to seek immediate care if symptoms worsen.

IMPORTANT:
1. **CRITICAL INVALID INPUT CHECK**:
   - If the input is just a greeting or nonsense, do not output the format.
2. **SCORING GUIDELINES**:
   - Be conservative. Use 90-100% for P1, 50-80% for P2, 30-50% for P3.
3. **LANGUAGE**: WRITE IN ENGLISH.
Severity: {severity}

Now provide the response in the EXACT format shown above."""
        
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                stop=["</s>", "<|im_end|>"]
            )
            
            english_result = response["choices"][0]["message"]["content"].strip()
            
            if "Priority 1:" not in english_result and "🥇" not in english_result:
                return self._fallback_response(symptoms, target_lang, specialist, severity, secondary_specialists)
            
            if target_lang != "en":
                return self._smart_translate_response(english_result, target_lang)
                
            return english_result
            
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._fallback_response(symptoms, target_lang, specialist, severity, secondary_specialists)

    def _smart_translate_response(self, text: str, target_lang: str) -> str:
        global medical_translator
        if not medical_translator or not medical_translator.available:
            return text
            
        lines = text.split('\n')
        translated_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                translated_lines.append("")
                continue
                
            if ("Priority" in line and "Score:" in line) or line.startswith("🥇") or line.startswith("🥈") or line.startswith("🥉"):
                translated_lines.append(line)
                continue
            
            prefix = ""
            content = line
            
            if line.startswith("Reasoning:"):
                prefix = "Reasoning: "
                content = line.replace("Reasoning:", "").strip()
            
            if content:
                trans_content = medical_translator.translate_from_english(content, target_lang)
                translated_lines.append(f"{prefix}{trans_content}")
            else:
                translated_lines.append(line)
                    
        return "\n".join(translated_lines)


    def _fallback_response(self, symptoms: str, language: str, specialist: str, severity: str, secondary_specialists: list = None) -> str:
        specs = [specialist]
        if secondary_specialists:
            for s in secondary_specialists:
                if s not in specs: specs.append(s)
        
        fallbacks = ["General Medicine", "Internal Medicine", "Family Physician"]
        for fb in fallbacks:
            if len(specs) >= 3: break
            if fb not in specs: specs.append(fb)
        
        specs = specs[:3]
        
        urgency = "Please seek immediate medical attention." if severity in ["emergency", "high"] else "Please schedule an appointment soon."
        
        base_response = f"""I understand you're experiencing these symptoms. {urgency}

🥇 Priority 1: {specs[0]} - Score: 90%
Reasoning: Based on your symptoms, this specialist is best equipped to evaluate and treat your condition. (Short explanation)

🥈 Priority 2: {specs[1]} - Score: 60%
Reasoning: This specialist can provide additional perspective and rule out related conditions.

🥉 Priority 3: {specs[2]} - Score: 30%
Reasoning: Consider this specialist for ongoing care and monitoring.

Remember to seek immediate care if symptoms worsen."""

        if language != "en":
            return self._smart_translate_response(base_response, language)
        
        return base_response
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        if not self.model or source_lang == "en":
            return text
        
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the following medical symptoms to English. Only output the translation, nothing else."},
                    {"role": "user", "content": text}
                ],
                max_tokens=128,
                temperature=0.3
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text


class SymptomNormalizer:
    
    NEGATION_PATTERNS = [
        r'\bno\s+((?:\w+\s*){1,3})',
        r'\bnot\s+having\s+((?:\w+\s*){1,3})',
        r'\bnot\s+((?:\w+\s*){1,3})',
        r"\bdon'?t\s+have\s+((?:\w+\s*){1,3})",
        r'\bwithout\s+((?:\w+\s*){1,3})',
        r'\bnever\s+((?:\w+\s*){1,3})',
        r'\babsent\s+((?:\w+\s*){1,3})',
        r'\bdenies\s+((?:\w+\s*){1,3})',
    ]
    
    def __init__(self):
        self.symptom_synonyms = {
            'heart attack': 'myocardial infarction chest pain',
            'stroke': 'cerebrovascular accident neurological deficit',
            'cannot breathe': 'severe respiratory distress dyspnea',
            'cant breathe': 'severe respiratory distress dyspnea',
            "can't breathe": 'severe respiratory distress dyspnea',
            'throwing up': 'vomiting nausea',
            'tummy ache': 'abdominal pain',
            'stomach ache': 'abdominal pain',
            'headache': 'cephalgia head pain',
            'dizzy': 'dizziness vertigo lightheadedness',
            'tired': 'fatigue exhaustion weakness',
            'short of breath': 'dyspnea shortness of breath',
        }
        
        self.MEDICAL_KEYWORDS = set()
        self._load_keywords()
    
    def detect_negations(self, text: str) -> dict:
        text_lower = text.lower()
        negated_symptoms = []
        
        for pattern in self.NEGATION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            negated_symptoms.extend(matches)
        
        remaining_text = text_lower
        for pattern in self.NEGATION_PATTERNS:
            remaining_text = re.sub(pattern, '', remaining_text)
        
        remaining_text = re.sub(r'[^\w\s]', '', remaining_text).strip()
        remaining_words = [w for w in remaining_text.split() if len(w) > 2]
        
        return {
            'has_negations': len(negated_symptoms) > 0,
            'negated_symptoms': negated_symptoms,
            'is_only_negation': len(remaining_words) == 0 and len(negated_symptoms) > 0,
            'remaining_text': ' '.join(remaining_words)
        }
    
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        for pattern, replacement in self.symptom_synonyms.items():
            if pattern in text:
                text = text.replace(pattern, replacement)
        
        return text.strip()
    

    def _load_keywords(self):
        json_path = os.path.join(BASE_DIR, "medical_keywords_clean.json")
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.MEDICAL_KEYWORDS = set(data.get("keywords", []))
                logger.info(f"✓ Loaded {len(self.MEDICAL_KEYWORDS)} medical keywords from JSON")
            else:
                logger.warning(f"⚠ medical_keywords_clean.json not found at {json_path}")
                self.MEDICAL_KEYWORDS = {
                    'pain', 'ache', 'fever', 'cough', 'doctor', 'hospital', 'medicine', 
                    'head', 'stomach', 'chest', 'back', 'leg', 'hand'
                }
        except Exception as e:
            logger.error(f"Error loading medical keywords: {e}")
            self.MEDICAL_KEYWORDS = {'pain', 'ache', 'fever'}

    NON_MEDICAL_WORDS = {
        'hello', 'hi', 'hey', 'bye', 'goodbye', 'thanks', 'thank', 'please',
        'homework', 'work', 'school', 'office', 'job', 'weather', 'food',
        'movie', 'music', 'game', 'play', 'sport', 'news', 'joke', 'story',
        'what', 'how', 'why', 'when', 'where', 'who', 'which', 'are', 'you'
    }
    
    def is_medical_query(self, text: str) -> dict:
        if not text or len(text.strip()) < 2:
            return {'is_medical': False, 'reason': 'Input too short', 'matched_keywords': []}
        
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))
        
        non_medical_match = words.intersection(self.NON_MEDICAL_WORDS)
        
        matched = words.intersection(self.MEDICAL_KEYWORDS)
        
        if any(ord(c) > 127 for c in text):
            is_non_ascii = True
            matched.add("non_ascii_fallback")
        else:
            is_non_ascii = False

        if len(non_medical_match) > 0 and len(matched) == 0:
            return {
                'is_medical': False,
                'reason': 'Non-medical query detected',
                'matched_keywords': []
            }
        
        is_medical = len(matched) > 0
        
        return {
            'is_medical': is_medical,
            'reason': 'Contains medical terms' if is_medical else 'No medical terms found',
            'matched_keywords': list(matched)
        }


class MedicalTriageSystem:
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.session = None
        self.tokenizer = None
        self.specialist_mapping = None
        self.config = None
        self.normalizer = SymptomNormalizer()
        
        self._load_components()
    
    def _load_components(self):
        logger.info(f"Loading medical triage system from: {self.model_dir}")
        
        model_path = os.path.join(self.model_dir, "model.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        specialists_csv = os.path.join(BASE_DIR, "specialists_grouped.csv")
        self.valid_specialists = set()
        if os.path.exists(specialists_csv):
            try:
                with open(specialists_csv, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'Specialist' in row:
                            self.valid_specialists.add(row['Specialist'].strip())
                logger.info(f"✓ Loaded {len(self.valid_specialists)} valid specialists from CSV")
            except Exception as e:
                logger.warning(f"Failed to load specialists CSV: {e}")
        else:
            logger.warning(f"specialists_grouped.csv not found at {specialists_csv}")

        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        logger.info(f"✓ Loaded ONNX model")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT",
                local_files_only=True
            )
            logger.info("✓ Loaded tokenizer (cached)")
        except Exception:
            logger.info("Downloading tokenizer from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        mapping_path = os.path.join(self.model_dir, "specialist_mapping.pkl")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.specialist_mapping = pickle.load(f)
            logger.info(f"✓ Loaded specialist mapping: {len(self.specialist_mapping)} entries")
        else:
            self.specialist_mapping = {}
        
        config_path = os.path.join(self.model_dir, "production_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"✓ Loaded {len(self.config.get('safety_rules', {}))} safety rules")
        else:
            self.config = {"safety_rules": {}, "severity_routing": {}}
        
    
    def refine_specialist(self, specialist: str) -> str:
        if not self.valid_specialists:
            return specialist
            
        if specialist in self.valid_specialists:
            return specialist
            
        spec_lower = specialist.lower()
        for valid in self.valid_specialists:
            if valid.lower() == spec_lower:
                return valid

        MANUAL_MAP = {
            "dermatology": "Dermatologist",
            "cardio": "Cardiologist",
            "general medicine": "General Practitioner",
            "ent": "ENT Specialist",
            "ortho": "Orthopedic Surgeon",
            "neuro": "Neurologist",
            "psych": "Psychiatrist",
            "internal medicine": "Internal Medicine Physician",
            "pediatrics": "Pediatrician",
            "gastro": "Gastroenterologist"
        }
        
        if spec_lower in MANUAL_MAP:
             return MANUAL_MAP[spec_lower]
        
        matches = difflib.get_close_matches(specialist, list(self.valid_specialists), n=1, cutoff=0.7)
        if matches:
            logger.info(f"Refined specialist '{specialist}' -> '{matches[0]}'")
            return matches[0]
            
        logger.warning(f"Specialist '{specialist}' not found in CSV. Defaulting to 'General Practitioner'.")
        return "General Practitioner"

    def check_safety_triggers(self, symptoms: str) -> Dict:
        norm_symptoms = symptoms.lower()
        triggered_rules = []
        max_severity = "low"
        forced_specialists = set()
        
        safety_rules = self.config.get("safety_rules", {})
        
        for symptom_pattern, rule in safety_rules.items():
            if symptom_pattern.lower() in norm_symptoms:
                triggered_rules.append({
                    'symptom': symptom_pattern,
                    'severity': rule.get('severity', 'moderate'),
                    'specialists': rule.get('specialists', [])
                })
                forced_specialists.update(rule.get('specialists', []))
                
                rule_severity = rule.get('severity', 'moderate')
                if rule_severity == 'emergency':
                    max_severity = 'emergency'
                elif rule_severity == 'high' and max_severity != 'emergency':
                    max_severity = 'high'
                elif rule_severity == 'moderate' and max_severity not in ['emergency', 'high']:
                    max_severity = 'moderate'
        
        return {
            'triggered': len(triggered_rules) > 0,
            'rules': triggered_rules,
            'max_severity': max_severity,
            'forced_specialists': list(forced_specialists)
        }
    
    def predict(self, symptom_text: str, return_details: bool = True) -> Dict:
        
        medical_check = self.normalizer.is_medical_query(symptom_text)
        
        if not medical_check['is_medical']:
            logger.info(f"Non-medical input detected: '{symptom_text[:50]}'")
            return {
                'primary_specialist': 'Not a medical query',
                'secondary_specialists': [],
                'severity': 'none',
                'confidence': 0.0,
                'urgency_message': 'Please describe your health symptoms or concerns. For example: "I have a headache" or "chest pain".',
                'emergency_routing': False,
                'red_flags': [],
                'input_symptoms': symptom_text,
                'is_medical_query': False,
                'reason': medical_check['reason']
            }
        
        negation_result = self.normalizer.detect_negations(symptom_text)
        
        if negation_result['is_only_negation']:
            logger.info(f"Detected only negated symptoms: {negation_result['negated_symptoms']}")
            return {
                'primary_specialist': 'No consultation needed',
                'secondary_specialists': [],
                'severity': 'none',
                'confidence': 100.0,
                'urgency_message': 'No symptoms detected. You appear to be healthy!',
                'emergency_routing': False,
                'red_flags': [],
                'input_symptoms': symptom_text,
                'negated_symptoms': negation_result['negated_symptoms'],
                'is_negation_only': True
            }
        
        normalized = self.normalizer.normalize(symptom_text)
        safety_result = self.check_safety_triggers(normalized)
        
        encoding = self.tokenizer(
            normalized,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        onnx_inputs = {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64),
        }
        
        input_names = [inp.name for inp in self.session.get_inputs()]
        if 'token_type_ids' in input_names:
            if 'token_type_ids' in encoding:
                onnx_inputs['token_type_ids'] = encoding['token_type_ids'].astype(np.int64)
            else:
                onnx_inputs['token_type_ids'] = np.zeros_like(encoding['input_ids']).astype(np.int64)
        
        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        unique_specialists = []
        seen_names = set()
        
        for idx in top_indices:
            confidence = float(probs[idx])
            raw_name = "General Medicine"
            
            if idx in self.specialist_mapping:
                spec_info = self.specialist_mapping[idx]
                if isinstance(spec_info, dict):
                    raw_name = spec_info.get('primary_specialist', 'General Medicine')
                else:
                    raw_name = str(spec_info)
            
            refined_name = self.refine_specialist(raw_name)
            
            if refined_name not in seen_names:
                seen_names.add(refined_name)
                unique_specialists.append({
                    'specialist': refined_name,
                    'confidence': confidence
                })
        
        if not unique_specialists:
            unique_specialists.append({'specialist': 'General Practitioner', 'confidence': 0.1})
            probs = np.array([0.1])
            top_indices = [0]

            
        if safety_result['triggered']:
            severity = safety_result['max_severity']
            
            forced_specs_refined = []
            if safety_result['forced_specialists']:
                for s in safety_result['forced_specialists']:
                    ref = self.refine_specialist(s)
                    if ref not in seen_names:
                        forced_specs_refined.append(ref)
                        seen_names.add(ref)
            
            if forced_specs_refined:
                primary_specialist = forced_specs_refined[0]
                candidates = forced_specs_refined[1:] + [s['specialist'] for s in unique_specialists]
                secondary_specialists = [s for s in candidates if s != primary_specialist]
                secondary_specialists = list(dict.fromkeys(secondary_specialists))[:2]
            else:
                primary_specialist = unique_specialists[0]['specialist']
                secondary_specialists = [s['specialist'] for s in unique_specialists[1:3]]
                
        else:
            severity = 'moderate' if probs[top_indices[0]] > 0.5 else 'low'
            primary_specialist = unique_specialists[0]['specialist']
            secondary_specialists = [s['specialist'] for s in unique_specialists[1:3]]
        
        severity_routing = self.config.get('severity_routing', {})
        urgency_message = severity_routing.get(severity, {}).get('message', 'Consult a healthcare provider')
        
        result = {
            'primary_specialist': primary_specialist,
            'secondary_specialists': secondary_specialists,
            'severity': severity,
            'urgency_message': urgency_message,
            'emergency_routing': severity in ['emergency', 'high'],
            'confidence': float(probs[top_indices[0]]) * 100,
        }
        
        if return_details:
            result['red_flags'] = [
                {'symptom': r['symptom'], 'reason': f"Triggers {r['severity']} priority"}
                for r in safety_result['rules']
            ]
            result['normalized_input'] = normalized
        
        return result


EDGE_TTS_VOICES = {
    'en': 'en-US-AriaNeural',
    'en-us': 'en-US-AriaNeural',
    'en-gb': 'en-GB-SoniaNeural',
    'hi': 'hi-IN-SwaraNeural',
    'te': 'te-IN-ShrutiNeural',
    'ta': 'ta-IN-PallaviNeural',
    'ml': 'ml-IN-SobhanaNeural',
    'kn': 'kn-IN-SapnaNeural',
    'bn': 'bn-IN-TanishaaNeural',
    'gu': 'gu-IN-DhwaniNeural',
    'mr': 'mr-IN-AarohiNeural',
    'pa': 'pa-IN-GurpreetNeural',
    'ur': 'ur-PK-UzmaNeural',
    'es': 'es-ES-ElviraNeural',
    'fr': 'fr-FR-DeniseNeural',
    'de': 'de-DE-KatjaNeural',
    'it': 'it-IT-ElsaNeural',
    'pt': 'pt-BR-FranciscaNeural',
    'ru': 'ru-RU-SvetlanaNeural',
    'zh': 'zh-CN-XiaoxiaoNeural',
    'zh-cn': 'zh-CN-XiaoxiaoNeural',
    'zh-tw': 'zh-TW-HsiaoChenNeural',
    'ja': 'ja-JP-NanamiNeural',
    'ko': 'ko-KR-SunHiNeural',
    'ar': 'ar-SA-ZariyahNeural',
    'tr': 'tr-TR-EmelNeural',
    'pl': 'pl-PL-ZofiaNeural',
    'nl': 'nl-NL-ColetteNeural',
    'sv': 'sv-SE-SofieNeural',
    'da': 'da-DK-ChristelNeural',
    'no': 'nb-NO-PernilleNeural',
    'fi': 'fi-FI-NooraNeural',
    'th': 'th-TH-PremwadeeNeural',
    'vi': 'vi-VN-HoaiMyNeural',
    'id': 'id-ID-GadisNeural',
    'ms': 'ms-MY-YasminNeural',
    'fil': 'fil-PH-BlessicaNeural',
    'uk': 'uk-UA-PolinaNeural',
    'cs': 'cs-CZ-VlastaNeural',
    'el': 'el-GR-AthinaNeural',
    'he': 'he-IL-HilaNeural',
    'ro': 'ro-RO-AlinaNeural',
    'hu': 'hu-HU-NoemiNeural',
    'sk': 'sk-SK-ViktoriaNeural',
    'bg': 'bg-BG-KalinaNeural',
    'hr': 'hr-HR-GabrijelaNeural',
    'sr': 'sr-RS-SophieNeural',
    'sl': 'sl-SI-PetraNeural',
    'lt': 'lt-LT-OnaNeural',
    'lv': 'lv-LV-EveritaNeural',
    'et': 'et-EE-AnuNeural',
    'af': 'af-ZA-AdriNeural',
    'sw': 'sw-KE-ZuriNeural',
    'am': 'am-ET-MekdesNeural',
    'ne': 'ne-NP-HemkalaNeural',
    'si': 'si-LK-ThiliniNeural',
    'my': 'my-MM-NilarNeural',
    'km': 'km-KH-SreymomNeural',
    'lo': 'lo-LA-KeomanyNeural',
}

class TTSEngine:
    
    SILERO_MODELS = {
        'ru': ('v3_1_ru', 'ru', 'baya'),
        'en': ('v3_en', 'en', 'en_0'),
        'de': ('v3_de', 'de', 'karlsson'),
        'es': ('v3_es', 'es', 'es_0'),
        'fr': ('v3_fr', 'fr', 'fr_0'),
        'ua': ('v3_ua', 'ua', 'mykyta'),
        'uz': ('v3_uz', 'uz', 'uz_0'),
        'xal': ('v3_xal', 'xal', 'xal_0'),
        'indic': ('v3_indic', 'indic', None),
    }
    
    def __init__(self, use_gtts=True):
        self.offline_engine = None
        self.silero_models = {}
        
        try:
            self.offline_engine = pyttsx3.init()
            self.offline_engine.setProperty('rate', 150)
            self.offline_engine.setProperty('volume', 0.9)
        except Exception as e:
            logger.warning(f"pyttsx3 init failed: {e}")
    
    def _get_edge_voice(self, lang: str) -> str:
        lang_lower = lang.lower()
        return EDGE_TTS_VOICES.get(lang_lower, EDGE_TTS_VOICES.get('en'))
    
    def _get_silero_model(self, lang: str):
        if not SILERO_AVAILABLE:
            return None, None, None
        
        lang_lower = lang.lower()
        
        if lang_lower in self.SILERO_MODELS:
            model_id, language, speaker = self.SILERO_MODELS[lang_lower]
        elif lang_lower in ['hi', 'te', 'ta', 'ml', 'kn', 'bn', 'gu', 'mr']:
            model_id, language, speaker = 'v3_indic', lang_lower, None
        else:
            return None, None, None
        
        if model_id not in self.silero_models:
            try:
                logger.info(f"Loading Silero model: {model_id}")
                model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language=language,
                    speaker=model_id
                )
                self.silero_models[model_id] = model
                logger.info(f"✓ Loaded Silero {model_id}")
            except Exception as e:
                logger.warning(f"Failed to load Silero {model_id}: {e}")
                return None, None, None
        
        return self.silero_models.get(model_id), language, speaker
    
    def _run_edge_tts_sync(self, text: str, output_path: str, voice: str):
        import concurrent.futures
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def generate():
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(output_path)
                loop.run_until_complete(generate())
            finally:
                loop.close()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            future.result(timeout=60)
    
    def convert(self, text: str, output_path: str, lang: str = 'en') -> str:
        
        if SILERO_AVAILABLE:
            try:
                model, language, speaker = self._get_silero_model(lang)
                if model is not None:
                    logger.info(f"Using Silero TTS for {lang}")
                    
                    audio = model.apply_tts(
                        text=text,
                        speaker=speaker,
                        sample_rate=48000
                    )
                    
                    wav_path = output_path.replace('.mp3', '.wav')
                    torchaudio.save(wav_path, audio.unsqueeze(0), 48000)
                    
                    from pydub import AudioSegment
                    audio_seg = AudioSegment.from_wav(wav_path)
                    audio_seg.export(output_path, format='mp3')
                    os.remove(wav_path)
                    
                    logger.info(f"✓ Generated TTS audio ({lang}) via Silero (offline)")
                    return output_path
            except Exception as e:
                logger.warning(f"Silero TTS failed: {e}")
        
        if EDGE_TTS_AVAILABLE:
            try:
                voice = self._get_edge_voice(lang)
                logger.info(f"Using edge-tts with voice: {voice}")
                
                import unicodedata
                clean_text = ''.join(
                    char for char in text 
                    if unicodedata.category(char) not in ('So', 'Sk', 'Sm')
                    and char not in '🔥🏆💪👨‍⚕️🩺💊🚨🚑⚕️❤️💙🟢🟡🔴📋✅❌⚠️🔍👋'
                )
                
                clean_text = clean_text.replace('**', '').strip()
                
                if not clean_text:
                    raise Exception("Text became empty after cleaning")
                
                logger.info(f"edge-tts text (cleaned): {clean_text[:50]}...")
                
                self._run_edge_tts_sync(clean_text, output_path, voice)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"✓ Generated TTS audio ({lang}) via edge-tts")
                    return output_path
                else:
                    raise Exception("edge-tts produced empty file")
            except Exception as e:
                logger.warning(f"edge-tts failed ({e})")
        
        if self.offline_engine:
            try:
                logger.info(f"Using pyttsx3 for {lang} (offline fallback)")
                self.offline_engine.save_to_file(text, output_path)
                self.offline_engine.runAndWait()
                logger.info(f"✓ Generated TTS audio ({lang}) via pyttsx3")
                return output_path
            except Exception as e:
                logger.error(f"pyttsx3 failed: {e}")
                raise e
        
        raise RuntimeError("No TTS engine available")


WHISPER_LANGUAGES = {
    'en': 'english', 'hi': 'hindi', 'te': 'telugu', 'ta': 'tamil', 
    'ml': 'malayalam', 'kn': 'kannada', 'bn': 'bengali', 'gu': 'gujarati',
    'mr': 'marathi', 'pa': 'punjabi', 'ur': 'urdu', 'ne': 'nepali',
    'si': 'sinhala', 'as': 'assamese', 'or': 'oriya',
    'es': 'spanish', 'fr': 'french', 'de': 'german', 'it': 'italian',
    'pt': 'portuguese', 'ru': 'russian', 'zh': 'chinese', 'ja': 'japanese',
    'ko': 'korean', 'ar': 'arabic', 'tr': 'turkish', 'pl': 'polish',
    'nl': 'dutch', 'sv': 'swedish', 'da': 'danish', 'no': 'norwegian',
    'fi': 'finnish', 'el': 'greek', 'he': 'hebrew', 'th': 'thai',
    'vi': 'vietnamese', 'id': 'indonesian', 'ms': 'malay', 'fil': 'tagalog',
    'uk': 'ukrainian', 'cs': 'czech', 'ro': 'romanian', 'hu': 'hungarian',
    'sk': 'slovak', 'bg': 'bulgarian', 'hr': 'croatian', 'sr': 'serbian',
    'sl': 'slovenian', 'lt': 'lithuanian', 'lv': 'latvian', 'et': 'estonian',
    'sw': 'swahili', 'af': 'afrikaans', 'zu': 'zulu', 'xh': 'xhosa',
    'am': 'amharic', 'my': 'myanmar', 'km': 'khmer', 'lo': 'lao',
    'ka': 'georgian', 'az': 'azerbaijani', 'uz': 'uzbek', 'kk': 'kazakh',
    'hy': 'armenian', 'is': 'icelandic', 'mk': 'macedonian', 'mt': 'maltese',
    'cy': 'welsh', 'ga': 'irish', 'eu': 'basque', 'gl': 'galician',
    'ca': 'catalan', 'la': 'latin', 'mn': 'mongolian', 'ps': 'pashto',
    'fa': 'persian', 'tl': 'tagalog', 'ha': 'hausa', 'yo': 'yoruba',
}

class STTEngine:
    
    def __init__(self, whisper_model: str = "base"):
        self.recognizer = sr.Recognizer()
        self.whisper_model = None
        self.vosk_model = None
        
        if WHISPER_AVAILABLE:
            try:
                logger.info(f"Loading Whisper model '{whisper_model}'...")
                self.whisper_model = whisper.load_model(whisper_model)
                logger.info(f"✓ Whisper '{whisper_model}' loaded (100+ languages)")
            except Exception as e:
                logger.warning(f"Whisper loading failed: {e}")
        
    
    def _convert_to_wav(self, audio_path: str) -> str:
        if audio_path.endswith('.wav'):
            return audio_path
        
        audio = AudioSegment.from_file(audio_path)
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format='wav')
        return wav_path
    
    def audio_to_text(self, audio_path: str, language: str = "en") -> str:
        wav_path = self._convert_to_wav(audio_path)
        
        if self.whisper_model:
            try:
                whisper_lang = WHISPER_LANGUAGES.get(language.split('-')[0].lower(), 'english')
                
                result = self.whisper_model.transcribe(
                    wav_path,
                    language=whisper_lang,
                    task="transcribe"
                )
                text = result.get("text", "").strip()
                if text:
                    logger.info(f"✓ Whisper STT ({whisper_lang}): {text[:50]}...")
                    return text
            except Exception as e:
                logger.warning(f"Whisper failed: {e}")
        
        if self.vosk_model and VOSK_AVAILABLE:
            try:
                import wave
                wf = wave.open(wav_path, "rb")
                rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
                rec.SetWords(True)
                
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    rec.AcceptWaveform(data)
                
                result = json.loads(rec.FinalResult())
                text = result.get("text", "").strip()
                wf.close()
                
                if text:
                    logger.info(f"✓ Vosk STT (offline): {text[:50]}...")
                    return text
            except Exception as e:
                logger.warning(f"Vosk failed: {e}")
        
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = self.recognizer.record(source)
                google_lang = language if '-' in language else f"{language}-{language.upper()}"
                if language == 'en':
                    google_lang = 'en-US'
                
                text = self.recognizer.recognize_google(audio_data, language=google_lang)
                logger.info(f"✓ Google STT ({google_lang}): {text[:50]}...")
                return text
        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Could not understand audio")
        except sr.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Speech recognition error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")


class ResponseFormatter:
    
    @staticmethod
    def format_text_response(result: Dict, include_ai_response: bool = False) -> str:
        
        response_parts = []
        
        if result['emergency_routing']:
            response_parts.append(f"⚠️ {result['urgency_message']}")
            response_parts.append(f"Severity: {result['severity'].upper()}")
        
        if result.get('red_flags'):
            response_parts.append("\nCritical symptoms detected:")
            for flag in result['red_flags']:
                response_parts.append(f"  • {flag['symptom']}")
        
        response_parts.append(f"\n🏥 Recommended: {result['primary_specialist']}")
        response_parts.append(f"Confidence: {result['confidence']:.1f}%")
        
        if result.get('secondary_specialists'):
            response_parts.append(f"Also consider: {', '.join(result['secondary_specialists'][:2])}")
        
        if include_ai_response and result.get('ai_response'):
            response_parts.append(f"\n💬 {result['ai_response']}")
        
        response_parts.append("\n⚠️ Disclaimer: Always consult qualified healthcare professionals.")
        
        return "\n".join(response_parts)


class TextInput(BaseModel):
    symptoms: str
    return_details: bool = True
    generate_response: bool = False

class MultilingualInput(BaseModel):
    symptoms: str
    generate_response: bool = True
    language: Optional[str] = None

class PredictionResponse(BaseModel):
    primary_specialist: str
    secondary_specialists: List[str]
    severity: str
    urgency_message: str
    emergency_routing: bool
    confidence: float
    language: Optional[str] = None
    ai_response: Optional[str] = None
    text_response: Optional[str] = None
    red_flags: Optional[List[Dict]] = None


app = FastAPI(
    title="Medical Triage System API",
    description="Multilingual medical triage with voice support (100+ languages)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

triage_system = None
language_detector = None
multilingual_brain = None
medical_translator = None
tts_engine = None
stt_engine = None
formatter = ResponseFormatter()

TEMP_DIR = tempfile.mkdtemp()


@app.on_event("startup")
async def startup_event():
    global triage_system, language_detector, multilingual_brain, medical_translator, tts_engine, stt_engine
    
    print("\n" + "="*80)
    print("MEDICAL TRIAGE API v3.0 - MULTILINGUAL")
    print("="*80)
    
    try:
        triage_system = MedicalTriageSystem(MODEL_DIR)
        print("✓ Medical Triage System (BioClinicalBERT ONNX)")
    except Exception as e:
        print(f"✗ Failed to load triage model: {e}")
        raise e
    
    try:
        language_detector = LanguageDetector(CACHE_DIR)
        print("✓ Language Detector (fastText, 176 languages)")
    except Exception as e:
        print(f"⚠ Language detector failed: {e}")
        language_detector = None
    
    try:
        medical_translator = MedicalTranslator(CACHE_DIR)
        if medical_translator.available:
            print("✓ Medical Translator (Google Translate - Unlimited)")
        else:
            print("⚠ Google Translator not available - using LLM fallback")
    except Exception as e:
        print(f"⚠ Translator failed: {e}")
        medical_translator = None
    
    try:
        tts_engine = TTSEngine(use_gtts=False)
        if EDGE_TTS_AVAILABLE:
            print("✓ TTS Engine (edge-tts Multilingual + pyttsx3 fallback)")
        else:
            print("✓ TTS Engine (pyttsx3 Offline)")
    except Exception as e:
        print(f"⚠ TTS engine failed: {e}")
        tts_engine = None
    
    try:
        multilingual_brain = MultilingualBrain(CACHE_DIR)
        print("✓ Multilingual Brain (Qwen2.5-1.5B-Instruct)")
    except Exception as e:
        print(f"⚠ Multilingual brain failed: {e}")
        multilingual_brain = None
    
    try:
        stt_engine = STTEngine(whisper_model="tiny")
        if WHISPER_AVAILABLE:
            print("✓ STT Engine (Whisper 100+ lang + Google fallback)")
        elif VOSK_AVAILABLE:
            print("✓ STT Engine (Vosk offline + Google fallback)")
        else:
            print("✓ STT Engine (Google Speech Recognition)")
    except Exception as e:
        print(f"⚠ STT engine failed: {e}")
        stt_engine = STTEngine.__new__(STTEngine)
        stt_engine.recognizer = sr.Recognizer()
        stt_engine.whisper_model = None
        stt_engine.vosk_model = None
    
    print("\n" + "="*80)
    print("OPEN SOURCE LIBRARIES (ALL FREE & UNLIMITED):")
    print("-" * 80)
    print(f"  TTS: edge-tts (75+ languages)  | Available: {EDGE_TTS_AVAILABLE}")
    print(f"  STT: Whisper (100+ languages)  | Available: {WHISPER_AVAILABLE}")
    print(f"  STT: Vosk (offline, 20+ lang)  | Available: {VOSK_AVAILABLE}")
    print(f"  Translation: deep-translator   | Available: {TRANSLATOR_AVAILABLE}")
    print(f"  Language: fastText (176 lang)  | Available: {FASTTEXT_AVAILABLE}")
    print(f"  LLM: llama-cpp (Qwen2.5-1.5B)  | Available: {LLAMA_CPP_AVAILABLE}")
    print("="*80)
    print("\n✓ API Ready - All Open Source, FREE, UNLIMITED")
    print("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    import shutil
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


@app.get("/")
async def root():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {
        "message": "Medical Triage API v3.0 - Multilingual",
        "ui": "Create index.html in the same directory to see the UI",
        "docs": "/docs"
    }


@app.get("/api")
async def api_info():
    return {
        "message": "Medical Triage API v3.0 - Multilingual",
        "models": {
            "triage": "BioClinicalBERT (ONNX)",
            "language_detection": "fastText (176 languages)",
            "multilingual_llm": "Qwen2.5-1.5B-Instruct"
        },
        "endpoints": {
            "ui": "/",
            "text": "/predict/text",
            "multilingual": "/predict/multi",
            "voice": "/predict/voice",
            "health": "/health"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "triage": triage_system is not None,
            "language_detector": language_detector is not None,
            "multilingual_brain": multilingual_brain is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    if not triage_system:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        detected_lang = "en"
        if language_detector:
            lang_result = language_detector.detect(input_data.symptoms)
            detected_lang = lang_result.get("language", "en")
        
        symptoms_for_model = input_data.symptoms
        if detected_lang != "en" and multilingual_brain:
            symptoms_for_model = multilingual_brain.translate_to_english(
                input_data.symptoms, detected_lang
            )
        
        result = triage_system.predict(symptoms_for_model, return_details=input_data.return_details)
        result['language'] = detected_lang
        
        if input_data.generate_response and multilingual_brain:
            result['ai_response'] = multilingual_brain.generate_response(
                symptoms=input_data.symptoms,
                language=detected_lang,
                specialist=result['primary_specialist'],
                secondary_specialists=result.get('secondary_specialists', []),
                severity=result['severity']
            )
        
        result['text_response'] = formatter.format_text_response(
            result, include_ai_response=input_data.generate_response
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multi", response_model=PredictionResponse)
async def predict_multilingual(input_data: MultilingualInput):
    if not triage_system:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        if input_data.language:
            detected_lang = input_data.language
            logger.info(f"Using user-specified language: {detected_lang}")
        elif language_detector:
            lang_result = language_detector.detect(input_data.symptoms)
            detected_lang = lang_result.get("language", "en")
            logger.info(f"Detected language: {detected_lang} (confidence: {lang_result.get('confidence', 0):.2f})")
        else:
            detected_lang = "en"
        
        symptoms_for_model = input_data.symptoms
        if detected_lang != "en":
            if medical_translator and medical_translator.available:
                symptoms_for_model = medical_translator.translate_to_english(
                    input_data.symptoms, detected_lang
                )
            elif multilingual_brain:
                symptoms_for_model = multilingual_brain.translate_to_english(
                    input_data.symptoms, detected_lang
                )
                logger.info(f"LLM translated to English: {symptoms_for_model[:100]}...")
        
        result = triage_system.predict(symptoms_for_model, return_details=True)
        result['language'] = detected_lang
        result['translated_symptoms'] = symptoms_for_model
        
        if multilingual_brain:
            if result.get('is_medical_query', True) is False:
                english_response = result.get('urgency_message', "Not a medical query.")
            else:
                primary = result['primary_specialist']
                secondaries = result.get('secondary_specialists', [])
                logger.info(f"Generating LLM response for: Primary='{primary}', Secondaries={secondaries}")
                
                english_response = multilingual_brain.generate_response(
                    symptoms=symptoms_for_model,
                    language=detected_lang,
                    specialist=primary,
                    secondary_specialists=secondaries,
                    severity=result['severity']
                )
            
            result['ai_response'] = english_response

        else:
            result['ai_response'] = f"Please consult a {result['primary_specialist']}."
        
        result['text_response'] = formatter.format_text_response(result, include_ai_response=True)
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/voice")
async def predict_voice(audio_file: UploadFile = File(...)):
    if not triage_system or not stt_engine:
        raise HTTPException(status_code=503, detail="Services not loaded")
    
    try:
        audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{audio_file.filename}")
        with open(audio_path, 'wb') as f:
            f.write(await audio_file.read())
        
        symptom_text = stt_engine.audio_to_text(audio_path)
        
        detected_lang = "en"
        if language_detector:
            lang_result = language_detector.detect(symptom_text)
            detected_lang = lang_result.get("language", "en")
        
        symptoms_for_model = symptom_text
        if detected_lang != "en" and multilingual_brain:
            symptoms_for_model = multilingual_brain.translate_to_english(symptom_text, detected_lang)
        
        result = triage_system.predict(symptoms_for_model, return_details=True)
        result['recognized_text'] = symptom_text
        result['language'] = detected_lang
        
        if multilingual_brain:
            result['ai_response'] = multilingual_brain.generate_response(
                symptoms=symptom_text,
                language=detected_lang,
                specialist=result['primary_specialist'],
                severity=result['severity']
            )
        
        result['text_response'] = formatter.format_text_response(result, include_ai_response=True)
        
        os.remove(audio_path)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/voice-to-voice")
async def predict_voice_to_voice(audio_file: UploadFile = File(...)):
    if not triage_system or not stt_engine or not tts_engine:
        raise HTTPException(status_code=503, detail="Services not loaded")
    
    try:
        input_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_input_{audio_file.filename}")
        with open(input_path, 'wb') as f:
            f.write(await audio_file.read())
        
        symptom_text = stt_engine.audio_to_text(input_path)
        
        detected_lang = "en"
        if language_detector:
            lang_result = language_detector.detect(symptom_text)
            detected_lang = lang_result.get("language", "en")
        
        symptoms_for_model = symptom_text
        if detected_lang != "en" and multilingual_brain:
            symptoms_for_model = multilingual_brain.translate_to_english(symptom_text, detected_lang)
        
        result = triage_system.predict(symptoms_for_model, return_details=True)
        
        if multilingual_brain:
            response_text = multilingual_brain.generate_response(
                symptoms=symptom_text,
                language=detected_lang,
                specialist=result['primary_specialist'],
                severity=result['severity']
            )
        else:
            response_text = formatter.format_text_response(result)
        
        output_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_output.mp3")
        tts_engine.convert(response_text, output_path, lang=detected_lang)
        
        os.remove(input_path)
        
        return FileResponse(
            output_path,
            media_type="audio/mpeg",
            filename="medical_advice.mp3",
            headers={
                "X-Recognized-Text": symptom_text,
                "X-Language": detected_lang,
                "X-Specialist": result['primary_specialist'],
                "X-Severity": result['severity']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-language")
async def detect_language(text: str = Form(...)):
    if not language_detector:
        raise HTTPException(status_code=503, detail="Language detector not loaded")
    
    result = language_detector.detect(text)
    return result


class TTSInput(BaseModel):
    text: str
    language: str = "en"

@app.post("/tts")
async def text_to_speech(input_data: TTSInput):
    if not tts_engine:
        raise HTTPException(status_code=503, detail="TTS not loaded")
    
    try:
        audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_tts.mp3")
        tts_engine.convert(input_data.text, audio_path, lang=input_data.language)
        return FileResponse(audio_path, media_type="audio/mpeg", filename="speech.mp3")
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stt")
async def speech_to_text(audio_file: UploadFile = File(...)):
    if not stt_engine:
        raise HTTPException(status_code=503, detail="STT not loaded")
    
    try:
        audio_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{audio_file.filename}")
        with open(audio_path, 'wb') as f:
            f.write(await audio_file.read())
        
        text = stt_engine.audio_to_text(audio_path)
        
        detected_lang = "en"
        if language_detector:
            lang_result = language_detector.detect(text)
            detected_lang = lang_result.get("language", "en")
        
        os.remove(audio_path)
        
        return {"recognized_text": text, "language": detected_lang}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 MEDICAL TRIAGE API v3.0 - MULTILINGUAL")
    print("="*80)
    print(f"\nModel Directory: {MODEL_DIR}")
    print(f"Cache Directory: {CACHE_DIR}")
    print("\nEndpoints:")
    print("  • /predict/text   - Text prediction (auto language)")
    print("  • /predict/multi  - Full multilingual with AI response")
    print("  • /predict/voice  - Voice to text prediction")
    print("  • /detect-language - Language detection only")
    print("\nAPI Documentation: http://localhost:8000/docs")
    print("="*80 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )