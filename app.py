# pip install transformers
# pip install streamlit
# npm install localtunnel

# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
# import streamlit as st

st.title("Neural Machine Translation")

source_lang_name = st.sidebar.text_area(label = ("Enter the source language (e.g., 'English'): ").strip(), max_chars = 20)
target_lang_name = st.sidebar.text_area(label = ("Enter the target language (e.g., 'Hindi'): ").strip(), max_chars = 20)

# Language to code mapping
language_code_map = {
    "Arabic": "ar_AR",
    "Czech": "cs_CZ",
    "German": "de_DE",
    "English": "en_XX",
    "Spanish": "es_XX",
    "Estonian": "et_EE",
    "Finnish": "fi_FI",
    "French": "fr_XX",
    "Hindi": "hi_IN",
    "Italian": "it_IT",
    "Japanese": "ja_XX",
    "Kazakh": "kk_KZ",
    "Korean": "ko_KR",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Burmese": "my_MM",
    "Nepali": "ne_NP",
    "Dutch": "nl_XX",
    "Romanian": "ro_RO",
    "Russian": "ru_RU",
    "Sinhala": "si_LK",
    "Turkish": "tr_TR",
    "Vietnamese": "vi_VN",
    "Chinese": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Bengali": "bn_IN",
    "Persian": "fa_IR",
    "Hebrew": "he_IL",
    "Croatian": "hr_HR",
    "Indonesian": "id_ID",
    "Georgian": "ka_GE",
    "Khmer": "km_KH",
    "Macedonian": "mk_MK",
    "Malayalam": "ml_IN",
    "Mongolian": "mn_MN",
    "Marathi": "mr_IN",
    "Polish": "pl_PL",
    "Pashto": "ps_AF",
    "Portuguese": "pt_XX",
    "Swedish": "sv_SE",
    "Swahili": "sw_KE",
    "Tamil": "ta_IN",
    "Telugu": "te_IN",
    "Thai": "th_TH",
    "Tagalog": "tl_XX",
    "Ukrainian": "uk_UA",
    "Urdu": "ur_PK",
    "Xhosa": "xh_ZA",
    "Galician": "gl_ES",
    "Slovene": "sl_SI"
}

# Look up language codes
source_lang = language_code_map.get(source_lang_name, None)
target_lang = language_code_map.get(target_lang_name, None)

# if not source_lang or not target_lang:
#     st.text("Invalid source or target language specified.")
# else:
#     sentence = st.sidebar.text_area(f"Enter your sentence in {source_lang_name} to get translation in {target_lang_name} ").strip()

# Load pre-trained model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

def translate(source_lang, target_lang, sentence):
    # Set the source language
    tokenizer.src_lang = source_lang

    # Encode the input sentence
    encoded_sentence = tokenizer(sentence, return_tensors="pt")

    # Generate translation
    generated_tokens = model.generate(
        **encoded_sentence,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )

    # Decode and return the translation
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# if source_lang_name and target_lang_name:
#   response = translate(source_lang, target_lang, sentence)
#   st.text(response)
sentence = st.sidebar.text_area(f"Enter your sentence in {source_lang_name} to get translation in {target_lang_name} ").strip()
try:
  if source_lang_name and target_lang_name:
    response = translate(source_lang, target_lang, sentence)
    st.text(response)
except:
  st.text("Invalid source or target language specified.")
