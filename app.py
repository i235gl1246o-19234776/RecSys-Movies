import streamlit as st
import joblib
import pickle
import pandas as pd

# Настройка страницы
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


# ==============================================================================
# Загрузка данных и моделей (кэшируется для ускорения)
# ==============================================================================
@st.cache_resource
def load_resources():
    # Загружаем обе модели
    catboost_model = joblib.load('catboost_ranker_model.pkl')
    svd_model = joblib.load('svd_model.pkl')

    with open('inference_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return catboost_model, svd_model, data


catboost_model, svd_model, data = load_resources()

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
# Вспомогательные функции
# ==============================================================================
def get_unseen_movies(user_id, train_df, movies_df):
    """Возвращает фильмы, которые пользователь еще не оценивал."""
    watched = set(train_df.loc[train_df.userId == user_id, "movieId"])
    return movies_df[~movies_df.movieId.isin(watched)].copy()


# ==============================================================================
# Логика рекомендаций: CatBoost
# ==============================================================================
def get_recommendations_catboost(user_id, top_n):
    candidates = get_unseen_movies(user_id, train_df, movies_df)

    # Мета-признаки
    candidates['year'] = candidates['title'].str.extract(r'\((\d{4})\)').astype(float)
    candidates['genres_count'] = candidates['genres'].str.split('|').apply(len)
    cand_genres = candidates['genres'].str.get_dummies('|')
    candidates = pd.concat([candidates, cand_genres], axis=1)

    # Статистики фильма
    candidates = candidates.merge(movie_stats, on='movieId', how='left')
    candidates['movie_mean'] = candidates['movie_mean'].fillna(global_movie_mean)
    candidates['movie_popularity'] = candidates['movie_popularity'].fillna(0)
    candidates['movie_rating_std'] = candidates['movie_rating_std'].fillna(0)

    # Статистики пользователя (Обработка Cold Start)
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

    for col in genre_columns:
        if col not in candidates.columns:
            candidates[col] = 0

    # Предсказание
    candidates_features = candidates[features]
    candidates['score'] = catboost_model.predict(candidates_features)

    # Форматирование результата
    result = (
        candidates.sort_values('score', ascending=False)
        [['movieId', 'title', 'score']]
        .head(top_n)
        .reset_index(drop=True)
    )
    result.insert(0, 'Rank', range(1, len(result) + 1))
    result = result.rename(columns={'title': 'Название фильма', 'score': 'Ranking Score'})

    return result, is_cold_start


# ==============================================================================
# Логика рекомендаций: SVD (Surprise)
# ==============================================================================
def get_recommendations_svd(user_id, top_n):
    candidates = get_unseen_movies(user_id, train_df, movies_df)

    # Проверка на Cold Start
    is_cold_start = user_id not in train_df['userId'].values

    if is_cold_start:
        # Fallback: сортировка по средней оценке фильма и популярности
        candidates = candidates.merge(movie_stats, on='movieId', how='left')
        candidates['movie_mean'] = candidates['movie_mean'].fillna(global_movie_mean)
        candidates['movie_popularity'] = candidates['movie_popularity'].fillna(0)

        result = (
            candidates
            .sort_values(['movie_mean', 'movie_popularity'], ascending=False)
            [['movieId', 'title', 'movie_mean']]
            .head(top_n)
            .reset_index(drop=True)
            .rename(columns={'movie_mean': 'Prediction Score'})
        )
    else:
        # Предсказываем рейтинги через SVD
        predictions = []
        for _, row in candidates.iterrows():
            movie_id = row['movieId']
            # Примечание: если при обучении Surprise Reader использовал строки,
            # здесь может понадобиться str(user_id) и str(movie_id)
            pred = svd_model.predict(user_id, movie_id)
            predictions.append({
                'movieId': movie_id,
                'title': row['title'],
                'Prediction Score': pred.est
            })

        result = (
            pd.DataFrame(predictions)
            .sort_values('Prediction Score', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    result.insert(0, 'Rank', range(1, len(result) + 1))
    result = result.rename(columns={'title': 'Название фильма'})

    return result, is_cold_start


# ==============================================================================
# Интерфейс Streamlit
# ==============================================================================
st.title("Персональная рекомендательная система (MovieLens)")
st.markdown("Сравнение подходов: **CatBoost Ranker** (ML-признаки) vs **SVD** (Matrix Factorization).")

# Боковая панель для ввода параметров
with st.sidebar:
    st.header("Параметры выдачи")

    min_user = int(train_df['userId'].min())
    max_user = int(train_df['userId'].max())

    user_id = st.number_input(
        "User ID",
        min_value=min_user,
        max_value=max_user + 1000,
        value=15,
        step=1,
        help=f"Диапазон обученных юзеров: {min_user} - {max_user}"
    )

    top_n = st.slider(
        "Количество фильмов (Top-N)",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )


    algo_choice = st.radio(
        "Выберите алгоритм рекомендаций:",
        options=["CatBoost Ranker", "SVD (Surprise)"],
        index=0,
        help="CatBoost использует мета-признаки и статистики. SVD основан на факторизации матрицы пользователь-фильм."
    )

    generate_btn = st.button("Сгенерировать рекомендации", use_container_width=True, type="primary")

# Основная область
if generate_btn:
    with st.spinner(f"Анализируем предпочтения с помощью {algo_choice}..."):
        if algo_choice == "CatBoost Ranker":
            recs, is_cold = get_recommendations_catboost(user_id, top_n)
        else:
            recs, is_cold = get_recommendations_svd(user_id, top_n)

    if is_cold:
        st.warning(
            f"**Cold Start:** Пользователь с ID **{user_id}** не найден в обучающей выборке. "
            f"{'Используются глобальные средние предпочтения.' if algo_choice == 'CatBoost Ranker' else 'Используется сортировка по популярности и средней оценке фильма.'}"
        )
    else:
        st.success(
            f"Рекомендации для пользователя **ID {user_id}** успешно сгенерированы алгоритмом **{algo_choice}**!")

    st.dataframe(
        recs,
        use_container_width=True,
        height=600,
        column_config={
            "Rank": st.column_config.NumberColumn("Место", width="small"),
            "Название фильма": st.column_config.TextColumn("Фильм", width="medium"),
            "Ranking Score": st.column_config.NumberColumn(
                "Score (CatBoost)",
                help="Скор модели ранжирования (чем выше, тем лучше)",
                format="%.4f",
                width="small"
            ) if algo_choice == "CatBoost Ranker" else None,
            "Prediction Score": st.column_config.NumberColumn(
                "Score (SVD)",
                help="Предсказанная оценка пользователя (scale 0.5 - 5.0)",
                format="%.2f",
                width="small"
            ) if algo_choice == "SVD (Surprise)" else None,
        }
    )
else:
    st.info("Выберите параметры в боковой панели, затем нажмите кнопку.")