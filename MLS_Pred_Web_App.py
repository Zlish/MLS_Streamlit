import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# GitHub CSV URLs
FIXTURES_URL = "https://raw.githubusercontent.com/Zlish/MLS_Streamlit/main/MLS_2025_Fixtures.csv"
TEAM_STATS_URL = "https://raw.githubusercontent.com/Zlish/MLS_Streamlit/main/MLS_Team_Stats.csv"

# ==============================
# Original helper functions
# ==============================
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

def add_features(df):
    df = df.sort_values('date').reset_index(drop=True)
    df['result'] = df.apply(
        lambda r: 'H' if r['home_goals'] > r['away_goals'] else
                  'A' if r['home_goals'] < r['away_goals'] else 'D',
        axis=1
    )

    teams = pd.unique(df[['home','away']].values.ravel('K'))
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
            'result': r['result']
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

def load_team_stats(url):
    ts = pd.read_csv(url)
    ts.columns = ts.columns.str.strip().str.lower()
    if 'squad' in ts.columns:
        ts = ts.rename(columns={'squad': 'team'})
    return ts

def merge_team_stats(feats_df, team_stats_df):
    merged = feats_df.merge(team_stats_df, left_on='home', right_on='team', how='left', suffixes=('', '_home'))
    merged.drop(columns=['team'], inplace=True)
    merged = merged.merge(team_stats_df, left_on='away', right_on='team', how='left', suffixes=('', '_away'))
    merged.drop(columns=['team'], inplace=True)
    return merged

def prepare_for_modeling(feats_df):
    df = feats_df.copy()
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['home_id'] = le_home.fit_transform(df['home'])
    df['away_id'] = le_away.fit_transform(df['away'])

    exclude_cols = ['home', 'away', 'result']
    feature_cols = [c for c in df.columns if (df[c].dtype in [np.float64, np.int64]) and (c not in exclude_cols)]
    feature_cols += ['home_id', 'away_id']
    
    X = df[feature_cols].fillna(0)
    y = df['result']
    return X, y, le_home, le_away

def train_and_eval_with_feature_importance(X, y, importance_threshold=0.01):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    class_weights = {'A':1.25, 'D':.33, 'H':2.5}
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=class_weights)
    clf.fit(X_train, y_train)

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    important_features = importances[importances >= importance_threshold].index.tolist()

    X_train_imp = X_train[important_features]
    X_test_imp = X_test[important_features]

    clf_imp = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    clf_imp.fit(X_train_imp, y_train)
    y_pred = clf_imp.predict(X_test_imp)
    scores = cross_val_score(clf_imp, X[important_features], y, cv=5, scoring='accuracy')
    
    report = classification_report(y_test, y_pred, output_dict=True)
    return clf_imp, important_features, report, scores.mean()

def predict_match(model, le_home, le_away, feats_df, selected_features, home_team, away_team, window=5):
    home_stats = feats_df[feats_df['home'] == home_team].tail(window)
    away_stats = feats_df[feats_df['away'] == away_team].tail(window)

    def avg_stat(df, stat_name):
        return df[stat_name].mean() if len(df) > 0 else 0.0

    home_gf_avg = avg_stat(home_stats, 'home_gf_avg')
    home_ga_avg = avg_stat(home_stats, 'home_ga_avg')
    home_form = avg_stat(home_stats, 'home_form')

    away_gf_avg = avg_stat(away_stats, 'away_gf_avg')
    away_ga_avg = avg_stat(away_stats, 'away_ga_avg')
    away_form = avg_stat(away_stats, 'away_form')

    home_id = le_home.transform([home_team])[0]
    away_id = le_away.transform([away_team])[0]

    base_features = {
        'home_id': home_id,
        'away_id': away_id,
        'home_gf_avg': home_gf_avg,
        'home_ga_avg': home_ga_avg,
        'home_form': home_form,
        'away_gf_avg': away_gf_avg,
        'away_ga_avg': away_ga_avg,
        'away_form': away_form
    }

    # Construct prediction DataFrame ensuring correct feature order and filling missing with 0
    X_pred = pd.DataFrame([[base_features.get(feat, 0.0) for feat in selected_features]],
                          columns=selected_features)

    pred_class = model.predict(X_pred)[0]
    pred_proba = model.predict_proba(X_pred)[0]
    return pred_class, dict(zip(model.classes_, pred_proba))

# ==============================
# Streamlit UI
# ==============================
st.title("⚽ MLS Match Prediction App")

@st.cache_data
def load_and_train():
    raw = pd.read_csv(FIXTURES_URL)
    clean = clean_fixtures_df(raw)
    feats = add_features(clean)
    team_stats = load_team_stats(TEAM_STATS_URL)
    merged_feats = merge_team_stats(feats, team_stats)
    X, y, le_home, le_away = prepare_for_modeling(merged_feats)
    model, selected_features, report, cv_score = train_and_eval_with_feature_importance(X, y)
    return merged_feats, model, selected_features, le_home, le_away, report, cv_score

merged_feats, model, selected_features, le_home, le_away, report, cv_score = load_and_train()

st.subheader("Model Performance")
st.write(f"**5-Fold CV Accuracy:** {cv_score:.3f}")
st.dataframe(pd.DataFrame(report).T)

team_list = sorted(list(pd.unique(merged_feats[['home', 'away']].values.ravel('K'))))
home_team = st.selectbox("Select Home Team", team_list)
away_team = st.selectbox("Select Away Team", team_list)

if st.button("Predict Match Result"):
    pred_class, pred_proba = predict_match(model, le_home, le_away, merged_feats, selected_features, home_team, away_team)
    st.success(f"Predicted Result: **{pred_class}**")
    st.write("Probabilities:")
    st.json(pred_proba)
