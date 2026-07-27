import sys
import re
import time
from pathlib import Path

import requests
import streamlit as st

# add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

BASE_URL = 'https://airline-recommendation-api.onrender.com'
API_URL = f'{BASE_URL}/predict'

# page configuration
st.set_page_config(
    page_title='Airline Recommendation Predictor',
    page_icon='✈️',
    layout='centered'
)

st.title('Airline Recommendation Predictor')
st.caption('Predict whether a passenger recommends the airline based on their review text.')

# initialize session state for text input
if 'review_text' not in st.session_state:
    st.session_state.review_text = ''


def clear_text():
    st.session_state.review_text = ''


# text area bound to session state with placeholder
review_input = st.text_area(
    'Enter review text:',
    placeholder='e.g., The flight was delayed for 3 hours, but the cabin crew was polite and helpful...',
    key='review_text'
)

# columns for action buttons
col_btn1, col_btn2 = st.columns([1, 5])

with col_btn1:
    predict_clicked = st.button('Predict', type='primary')

with col_btn2:
    st.button('Clear', on_click=clear_text)


def wake_up_api() -> bool:
    """Агресивне пробудження Render: робить до 6 спроб з павзами"""
    with st.spinner("🚀 Waking up Render server... This can take 30-50 seconds on Free Tier."):

        for attempt in range(6):
            try:

                wake_response = requests.get(f"{BASE_URL}/health", timeout=10)
                if wake_response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:

                pass

            time.sleep(7)

    return False


# prediction logic
if predict_clicked:
    cleaned_input = review_input.strip()

    if not cleaned_input:
        st.warning('Please enter a review.')

    else:
        if re.search(r'[\u0400-\u04FF]', cleaned_input):
            st.toast(
                '⚠️ Warning: Cyrillic characters detected. Please enter review in English.',
                icon='⚠️'
            )

        try:
            # 1. Пробуджуємо бекенд з анімацією
            with st.spinner('🚀 Connecting to API (waking up server if asleep)...'):
                api_ready = wake_up_api()

            if not api_ready:
                st.error('The server is taking too long to wake up. Please click "Predict" again in a few seconds.')
            else:
                # 2. Робимо реальний запит для передбачення
                response = requests.post(
                    API_URL,
                    json={'review_text': cleaned_input},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()

                    prediction = data['prediction']
                    confidence = data['confidence']
                    top_features = data['top_features']

                    st.markdown('---')

                    col_res, col_words = st.columns(2)

                    with col_res:
                        st.subheader('Result')

                        if prediction == 1:
                            st.success('**Prediction:** Recommended')
                        else:
                            st.error('**Prediction:** Not Recommended')

                        st.metric(
                            label='Model Confidence',
                            value=f'{confidence * 100:.1f}%'
                        )

                    with col_words:
                        st.subheader('Most Influential Words')

                        if top_features:
                            for word, _ in top_features:
                                st.markdown(f'- **{word}**')
                        else:
                            st.info(
                                'Text contains unknown or uninformative words for analysis.'
                            )

                else:
                    st.error(
                        f'API returned status code {response.status_code}. '
                        'Please check the FastAPI server.'
                    )

        except requests.exceptions.Timeout:
            st.error('The request timed out. Please try again.')

        except requests.exceptions.ConnectionError:
            st.error('Could not connect to FastAPI server.')

        except Exception as e:
            st.error(f'Unexpected error: {e}')