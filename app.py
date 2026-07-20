import streamlit as st
import joblib
import pickle
import pandas as pd
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# ==============================================================================
# Загрузка данных и модели (кэшируется для ускорения)
# ==============================================================================
@st.cache_resource
def load_resources():
    model = joblib.load('catboost_ranker_model.pkl')
    with open('inference_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return model, data


model, data = load_resources()

# Извлекаем нужные словари из загруженных данных
movies_df = data['movies']
train_df = data['train_df']
movie_stats = data['movie_stats']
user_stats = data['user_stats']
global_movie_mean = data['global_movie_mean']
global_user_mean = data['global_user_mean']
genre_columns = data['genre_columns']
features = data['features']


# ==============================================================================
# Логика рекомендаций (адаптирована из ноутбука)
# ==============================================================================
def get_recommendations(user_id, top_n):
    # 1. Фильмы, которые юзер еще не смотрел
    watched = set(train_df.loc[train_df.userId == user_id, "movieId"])
    candidates = movies_df[~movies_df.movieId.isin(watched)].copy()

    # 2. Мета-признаки
    candidates['year'] = candidates['title'].str.extract(r'\((\d{4})\)').astype(float)
    candidates['genres_count'] = candidates['genres'].str.split('|').apply(len)
    cand_genres = candidates['genres'].str.get_dummies('|')
    candidates = pd.concat([candidates, cand_genres], axis=1)

    # 3. Статистики фильма
    candidates = candidates.merge(movie_stats, on='movieId', how='left')
    candidates['movie_mean'] = candidates['movie_mean'].fillna(global_movie_mean)
    candidates['movie_popularity'] = candidates['movie_popularity'].fillna(0)
    candidates['movie_rating_std'] = candidates['movie_rating_std'].fillna(0)

    # 4. Статистики пользователя (Обработка Cold Start)
    user_info = user_stats[user_stats.userId == user_id]
    is_cold_start = user_info.empty

    if not is_cold_start:
        candidates['user_mean'] = user_info['user_mean'].values[0]
        candidates['user_activity'] = user_info['user_activity'].values[0]
        candidates['user_rating_std'] = user_info['user_rating_std'].values[0]
    else:
        candidates['user_mean'] = global_user_mean
        candidates['user_activity'] = 0
        candidates['user_rating_std'] = 0

    candidates['mean_difference'] = candidates['user_mean'] - candidates['movie_mean']

    # Выравниваем колонки жанров
    for col in genre_columns:
        if col not in candidates.columns:
            candidates[col] = 0

    # 5. Предсказание скором
    candidates_features = candidates[features]
    candidates['score'] = model.predict(candidates_features)

    # 6. Сортировка и возврат
    result = (
        candidates.sort_values('score', ascending=False)
        [['movieId', 'title', 'score']]
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index = result.index + 1  # Нумерация с 1
    result.index.name = 'Rank'
    result = result.rename(columns={'title': 'Название фильма', 'score': 'Ranking Score'})

    return result, is_cold_start


# ==============================================================================
# Интерфейс Streamlit
# ==============================================================================
st.title("🎬 Персональная рекомендательная система (MovieLens)")
st.markdown("Система на базе **CatBoost Ranker** с учетом временного сплита и статистик.")

# Боковая панель для ввода параметров
with st.sidebar:
    st.header("⚙️ Параметры выдачи")

    # Получаем минимальный и максимальный ID юзера для подсказки
    min_user = int(train_df['userId'].min())
    max_user = int(train_df['userId'].max())

    user_id = st.number_input(
        "👤 User ID",
        min_value=min_user,
        max_value=max_user + 1000,  # Разрешаем ввод несуществующих для теста Cold Start
        value=15,
        step=1,
        help=f"Диапазон обученных юзеров: {min_user} - {max_user}"
    )

    top_n = st.slider(
        "📊 Количество фильмов (Top-N)",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    generate_btn = st.button("🚀 Сгенерировать рекомендации", use_container_width=True, type="primary")

# Основная область
if generate_btn:
    with st.spinner("Анализируем предпочтения и генерируем Top-K..."):
        recs, is_cold = get_recommendations(user_id, top_n)

    if is_cold_start:
        st.warning(
            f"⚠️ **Cold Start:** Пользователь с ID **{user_id}** не найден в обучающей выборке. Используются глобальные средние предпочтения.")
    else:
        st.success(f"✅ Рекомендации для пользователя **ID {user_id}** успешно сгенерированы!")

    st.dataframe(
        recs,
        use_container_width=True,
        height=600,
        column_config={
            "Ranking Score": st.column_config.NumberColumn(
                "Ranking Score",
                help="Скор модели ранжирования (чем выше, тем больше вероятность, что фильм понравится)",
                format="%.4f"
            )
        }
    )
else:
    st.info("👈 Выберите User ID и количество фильмов в боковой панели, затем нажмите кнопку.")