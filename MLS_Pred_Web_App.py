import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------- Data Cleaning ----------
def clean_fixtures_df(df):
    df = df.copy()
    if 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    else:
        raise ValueError("No 'Date' column found in CSV")

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "home" in lc:
            col_map[c] = "home"
        elif "away" in lc:
            col_map[c] = "away"
        elif "score" in lc:
            col_map[c] = "score"
    df.rename(columns=col_map, inplace=True)

    if 'score' in df.columns:
        def parse_score(s):
            try:
                s = str(s)
                for sep in ['-', '–', '—']:
                    if sep in s:
                        h, a = s.split(sep)
                        return int(h.strip()), int(a.strip())
                return np.nan, np.nan
            except:
                return np.nan, np.nan
        scores = df['score'].apply(parse_score)
        df['home_goals'] = scores.apply(lambda x: x[0])
        df['away_goals'] = scores.apply(lambda x: x[1])

    df = df.dropna(subset=['home', 'away', 'home_goals', 'away_goals']).reset_index(drop=True)
    return df

# ---------- Feature Engineering ----------
def add_features(df):
    df = df.sort_values('date').reset_index(drop=True)
    teams = pd.unique(df[['home', 'away']].values.ravel('K'))
    team_history = {t: [] for t in teams}
    rows = []
    window = 5

    for _, r in df.iterrows():
        home = r['home']
        away = r['away']

        def team_agg(team):
            hist = team_history[team][-window:]
            if not hist:
                return {'gf_avg': 0.0, 'ga_avg': 0.0, 'form_points': 0.0}
            gf = np.mean([m['gf'] for m in hist])
            ga = np.mean([m['ga'] for m in hist])
            form = np.mean([m['points'] for m in hist])
            return {'gf_avg': gf, 'ga_avg': ga, 'form_points': form}

        home_agg = team_agg(home)
        away_agg = team_agg(away)

        rows.append({
            'home': home, 'away': away,
            'home_gf_avg': home_agg['gf_avg'], 'home_ga_avg': home_agg['ga_avg'], 'home_form': home_agg['form_points'],
            'away_gf_avg': away_agg['gf_avg'], 'away_ga_avg': away_agg['ga_avg'], 'away_form': away_agg['form_points'],
            'home_goals': r['home_goals'],
            'away_goals': r['away_goals']
        })

        hg, ag = r['home_goals'], r['away_goals']
        if hg > ag:
            hpts, apts = 3, 0
        elif hg < ag:
            hpts, apts = 0, 3
        else:
            hpts, apts = 1, 1
        team_history[home].append({'gf': hg, 'ga': ag, 'points': hpts})
        team_history[away].append({'gf': ag, 'ga': hg, 'points': apts})

    return pd.DataFrame(rows)

# ---------- Team Stats ----------
def load_team_stats(df):
    df.columns = df.columns.str.strip().str.lower()
    if 'squad' in df.columns:
        df = df.rename(columns={'squad': 'team'})
    return df

def merge_team_stats(feats_df, team_stats_df):
    merged = feats_df.merge(team_stats_df, left_on='home', right_on='team', how='left', suffixes=('', '_home'))
    merged.drop(columns=['team'], inplace=True)
    merged = merged.merge(team_stats_df, left_on='away', right_on='team', how='left', suffixes=('', '_away'))
    merged.drop(columns=['team'], inplace=True)
    return merged

# ---------- Modeling ----------
def prepare_for_modeling(feats_df):
    df = feats_df.copy()
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['home_id'] = le_home.fit_transform(df['home'])
    df['away_id'] = le_away.fit_transform(df['away'])

    exclude_cols = ['home', 'away', 'home_goals', 'away_goals']
    feature_cols = [c for c in df.columns if (df[c].dtype in [np.float64, np.int64]) and (c not in exclude_cols)]
    feature_cols += ['home_id', 'away_id']

    X = df[feature_cols].fillna(0)
    y = df['home_goals'] + df['away_goals']

    return X, y, le_home, le_away

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, rmse

def predict_total_goals(model, le_home, le_away, feats_df, selected_features, home_team, away_team, window=5):
    home_stats = feats_df[feats_df['home'] == home_team].tail(window)
    away_stats = feats_df[feats_df['away'] == away_team].tail(window)

    def avg_stat(df, stat_name):
        return df[stat_name].mean() if len(df) > 0 else 0.0

    base_features = {}
    try:
        base_features['home_id'] = le_home.transform([home_team])[0]
        base_features['away_id'] = le_away.transform([away_team])[0]
        base_features['home_gf_avg'] = avg_stat(home_stats, 'home_gf_avg')
        base_features['home_ga_avg'] = avg_stat(home_stats, 'home_ga_avg')
        base_features['home_form'] = avg_stat(home_stats, 'home_form')
        base_features['away_gf_avg'] = avg_stat(away_stats, 'away_gf_avg')
        base_features['away_ga_avg'] = avg_stat(away_stats, 'away_ga_avg')
        base_features['away_form'] = avg_stat(away_stats, 'away_form')
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None

    pred_dict = {feat: 0.0 for feat in selected_features}
    pred_dict.update(base_features)

    X_pred = pd.DataFrame([pred_dict])
    X_pred = X_pred[selected_features]

    pred_total_goals = model.predict(X_pred)[0]
    pred_total_goals = max(0, round(pred_total_goals))
    return pred_total_goals

# ---------- Streamlit App ----------
def main():
    st.title("MLS Match Total Goals Predictor")

    # Load directly from GitHub
    fixtures_url = "https://raw.githubusercontent.com/Zlish/MLS_Streamlit/main/MLS_2025_Fixtures.csv"
    team_stats_url = "https://raw.githubusercontent.com/Zlish/MLS_Streamlit/main/MLS_Team_Stats.csv"

    try:
        raw_fixtures = pd.read_csv(fixtures_url)
        raw_team_stats = pd.read_csv(team_stats_url)

        clean = clean_fixtures_df(raw_fixtures)
        feats = add_features(clean)
        team_stats = load_team_stats(raw_team_stats)
        merged_feats = merge_team_stats(feats, team_stats)

        X, y, le_home, le_away = prepare_for_modeling(merged_feats)
        model, rmse = train_model(X, y)

        st.write(f"Model trained. RMSE on test set: {rmse:.3f}")

        teams = sorted(list(pd.unique(merged_feats[['home', 'away']].values.ravel('K'))))
        home_team = st.selectbox("Select Home Team:", teams)
        away_team = st.selectbox("Select Away Team:", teams)

        if st.button("Predict Total Goals"):
            pred_goals = predict_total_goals(model, le_home, le_away, merged_feats, X.columns.tolist(), home_team, away_team)
            if pred_goals is not None:
                st.success(f"Predicted total goals for {home_team} vs {away_team}: {pred_goals}")

    except Exception as e:
        st.error(f"Error processing data: {e}")

if __name__ == "__main__":
    main()
